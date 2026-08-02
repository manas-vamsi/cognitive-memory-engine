"""The CME facade — the plug-and-play surface an LLM sits behind.

Wires the registry and all seven engines together so a caller does not have to
know they exist. The three operations a reasoning client actually needs:

    ingest(text)   — learn from a document
    context(query) — the best grounded memories to put in a prompt
    verify(answer) — check what the model said against what is actually known
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from cme_python.config import settings
from cme_python.engines.belief import (
    BeliefEngine,
    Extractor,
    LLMExtractor,
    rule_based_extract,
)
from cme_python.engines.entailment import get_detector
from cme_python.engines.evidence import EvidenceEngine, GroundingReport, Justification
from cme_python.engines.graph import KnowledgeGraph
from cme_python.engines.memory import DEFAULT_HALF_LIFE_DAYS, MemoryEngine, MemoryStats
from cme_python.engines.optimization import OptimizationEngine
from cme_python.engines.quantum_layer import get_solver
from cme_python.engines.reasoning import Contradiction, ReasoningEngine, Resolution
from cme_python.engines.vectors import VectorRetriever, cache_path_for
from cme_python.models import Belief, MemoryTier, Revision, SourceKind
from cme_python.store import open_store


class GroundedContext(BaseModel):
    """Memories chosen for a query, each with the proof behind it."""

    query: str
    beliefs: list[Belief]
    justifications: list[Justification]
    tokens: int

    def as_prompt(self) -> str:
        """The block to paste in front of a model, evidence included."""
        if not self.beliefs:
            return ""
        lines = ["Known facts, with confidence and source:"]
        for belief, why in zip(self.beliefs, self.justifications, strict=True):
            sources = ", ".join(e.locator or str(e.source) for e in why.supporting)
            lines.append(
                f"- {belief.statement} "
                f"({belief.confidence:.0%} confident{'; ' + sources if sources else ''})"
            )
        return "\n".join(lines)


def _extractor(name: str | None) -> Extractor:
    """The claim extractor to hand the Belief Engine.

    Rule-based unless asked otherwise, and imported lazily so the default path
    never touches a vendor SDK. A model that cannot be built is a hard error
    here rather than a quiet downgrade: a caller who asked for LLM extraction
    and silently got regexes would have no way to know why their documents came
    out thin.
    """
    if (name or settings.extractor) != "llm":
        return rule_based_extract

    from cme_python.clients.base import build_client  # noqa: PLC0415

    if not settings.llm:
        raise ValueError("CME_EXTRACTOR=llm needs CME_LLM set to 'claude' or 'openai'.")
    return LLMExtractor(build_client(settings.llm, settings.llm_model))


class Maintenance(BaseModel):
    """What one maintenance pass did to the registry."""

    decayed: int
    """Beliefs that faded for want of reinforcement."""
    weakened: int
    """Beliefs a better-backed contradiction pushed down."""
    retired: int
    """Beliefs superseded outright. Out of recall, still on record."""
    pruned: int
    """Beliefs deleted for being disproven into irrelevance."""

    @property
    def changed(self) -> int:
        return self.decayed + self.weakened + self.retired + self.pruned


class CME:
    """A persistent brain for a reasoning client."""

    def __init__(
        self,
        database: str | None = None,
        *,
        solver: str | None = None,
        retrieval: str | None = None,
        detector: str | None = None,
        extractor: str | None = None,
    ) -> None:
        self.store = open_store(database or settings.database)
        self.beliefs = BeliefEngine(self.store, extractor=_extractor(extractor))
        mode = retrieval or settings.retrieval
        self._vectors = (
            VectorRetriever(self.store, cache=cache_path_for(self.store.path))
            if mode == "vector"
            else None
        )
        self.evidence = EvidenceEngine(self.store, retriever=self._vectors)
        self.optimizer = OptimizationEngine(
            self.evidence, solver=get_solver(solver or settings.solver)
        )
        self.reasoning = ReasoningEngine(
            self.store, detector=get_detector(detector or settings.detector, **self._embedder())
        )
        self.memory = MemoryEngine(self.store)

    def _embedder(self) -> dict:
        """The embedding function to hand a detector, if there is one.

        Only vector retrieval builds an embedder, and a remote index may not
        expose one at all. A detector that wants embeddings to pick candidate
        pairs decides for itself what to do without — falling back to every
        pair is a reasonable answer at small sizes, and better than this
        constructor failing over an optional accelerator.
        """
        index = getattr(self._vectors, "index", None)
        embedder = getattr(index, "embedder", None)
        return {"embed": embedder.embed} if embedder is not None else {}

    def __enter__(self) -> CME:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        # Shutdown is the right moment to write the vector cache: saving on
        # every change would cost more than the rebuild it avoids.
        if self._vectors is not None:
            self._vectors.persist()
        self.store.close()

    @property
    def graph(self) -> KnowledgeGraph:
        return self.reasoning.graph

    # --- the three operations ---------------------------------------------

    def ingest(
        self,
        text: str,
        *,
        source: SourceKind = SourceKind.UNKNOWN,
        locator: str | None = None,
        connections: Iterable[str] = (),
        tier: MemoryTier = MemoryTier.GENERAL,
        scope: str | None = None,
    ) -> list[Belief]:
        """Learn from a document. Re-ingesting reinforces rather than duplicates."""
        filed = self.beliefs.ingest(text, source=source, locator=locator, connections=connections)
        return self.memory.remember(filed, tier=tier, scope=scope)

    def context(
        self,
        query: str,
        *,
        budget: float | None = None,
        tier: MemoryTier | None = None,
        scope: str | None = None,
    ) -> GroundedContext:
        """The best set of memories for a query, within a token budget.

        Not a top-k slice: the Optimization Engine trades relevance against
        redundancy so the budget buys distinct facts rather than the same one
        three times. `tier` and `scope` confine recall to one body of memory.
        """
        chosen = self.optimizer.select(
            query,
            budget=budget or settings.context_budget,
            within=self.memory.view(tier, scope) if tier or scope else None,
        )
        return GroundedContext(
            query=query,
            beliefs=chosen,
            justifications=[self.evidence.justify(b) for b in chosen],
            tokens=sum(self.optimizer.cost(b) for b in chosen),
        )

    def verify(
        self,
        answer: str,
        *,
        tier: MemoryTier | None = None,
        scope: str | None = None,
    ) -> GroundingReport:
        """Check generated text against the registry, claim by claim.

        Verifying inside the same slice the answer was drawn from matters: a
        claim backed only by another tier is not backed for this caller.
        """
        return self.evidence.ground(
            answer, within=self.memory.view(tier, scope) if tier or scope else None
        )

    def stats(self) -> MemoryStats:
        return self.memory.stats()

    def decay(self, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> int:
        """Age beliefs nothing has reinforced. Returns how many moved.

        Run it on a schedule if you want one; nothing runs it for you. The
        facade returns a count rather than the confidences because the caller
        that wants the detail already has `cme.memory.decay()`.
        """
        return len(self.memory.decay(half_life_days=half_life_days))

    def maintain(
        self,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        prune: bool = True,
    ) -> Maintenance:
        """One upkeep pass: age, reconcile, then clear out the disproven.

        The three background operations are individually opt-in, which left a
        deployment with no single thing to put on a schedule. This is that
        thing. Run it nightly, or never — nothing runs it for you, because
        rewriting somebody's memory unasked is not a default.

        The order is the argument. Ageing first means contradictions are judged
        on current confidences rather than stale ones, so a claim nobody has
        backed in a year no longer outranks a fresh one purely for having been
        written first. Pruning last means anything the first two steps
        disproved leaves in the same pass instead of lingering until the next.

        Note that ageing cannot itself cause a deletion: decay stops at a floor
        well above the pruning threshold, so only contradicting evidence can
        push a belief far enough down to be removed.
        """
        decayed = len(self.memory.decay(half_life_days=half_life_days))
        resolved = self.reasoning.reconcile()
        return Maintenance(
            decayed=decayed,
            weakened=sum(1 for r in resolved if not r.retired),
            retired=sum(1 for r in resolved if r.retired),
            pruned=self.memory.prune() if prune else 0,
        )

    # --- inspection --------------------------------------------------------

    def explain(self, belief_id: str) -> Justification | None:
        belief = self.store.get(belief_id)
        return self.evidence.justify(belief) if belief else None

    def timeline(self, belief_id: str) -> list[Revision]:
        """Every change to a belief's confidence, oldest first.

        The answer to "has this been checked lately, or has it just been sitting
        there?" — which a bare confidence score cannot tell you.
        """
        return self.memory.timeline(belief_id)

    def supersede(self, old_id: str, new_id: str) -> Belief | None:
        """Retire a belief in favour of a better-sourced one that replaces it."""
        return self.memory.supersede(old_id, new_id)

    def contradictions(self) -> list[Contradiction]:
        return self.reasoning.contradictions()

    def reconcile(self) -> list[Resolution]:
        """Act on every contradiction: retire the decisively beaten, weaken the rest.

        Detection alone leaves the registry knowingly inconsistent — able to say
        two beliefs cannot both be true, and serving both regardless. Opt-in,
        like `decay`: acting on somebody's memory is their call, not ours.
        """
        return self.reasoning.reconcile()
