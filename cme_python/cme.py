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
from cme_python.engines.belief import BeliefEngine
from cme_python.engines.evidence import EvidenceEngine, GroundingReport, Justification
from cme_python.engines.graph import KnowledgeGraph
from cme_python.engines.optimization import OptimizationEngine
from cme_python.engines.quantum_layer import get_solver
from cme_python.engines.reasoning import Contradiction, ReasoningEngine
from cme_python.models import Belief, SourceKind
from cme_python.store import BeliefStore


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


class CME:
    """A persistent brain for a reasoning client."""

    def __init__(
        self,
        database: str | None = None,
        *,
        solver: str | None = None,
    ) -> None:
        self.store = BeliefStore(database or settings.database)
        self.beliefs = BeliefEngine(self.store)
        self.evidence = EvidenceEngine(self.store)
        self.optimizer = OptimizationEngine(
            self.evidence, solver=get_solver(solver or settings.solver)
        )
        self.reasoning = ReasoningEngine(self.store)

    def __enter__(self) -> CME:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
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
    ) -> list[Belief]:
        """Learn from a document. Re-ingesting reinforces rather than duplicates."""
        return self.beliefs.ingest(text, source=source, locator=locator, connections=connections)

    def context(self, query: str, *, budget: float | None = None) -> GroundedContext:
        """The best set of memories for a query, within a token budget.

        Not a top-k slice: the Optimization Engine trades relevance against
        redundancy so the budget buys distinct facts rather than the same one
        three times.
        """
        chosen = self.optimizer.select(query, budget=budget or settings.context_budget)
        return GroundedContext(
            query=query,
            beliefs=chosen,
            justifications=[self.evidence.justify(b) for b in chosen],
            tokens=sum(self.optimizer.cost(b) for b in chosen),
        )

    def verify(self, answer: str) -> GroundingReport:
        """Check generated text against the registry, claim by claim."""
        return self.evidence.ground(answer)

    # --- inspection --------------------------------------------------------

    def explain(self, belief_id: str) -> Justification | None:
        belief = self.store.get(belief_id)
        return self.evidence.justify(belief) if belief else None

    def contradictions(self) -> list[Contradiction]:
        return self.reasoning.contradictions()
