"""Reasoning Engine — inference over the belief graph.

Three jobs from the brief: multi-hop reasoning (A → B → C), contradiction
detection (flagging beliefs that clash), and belief propagation (when one
belief's confidence moves, everything resting on it moves too).
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

from cme_python.config import settings
from cme_python.engines.evidence import tokenise
from cme_python.engines.graph import BELIEF, CONCEPT, KnowledgeGraph, belief_node
from cme_python.engines.native import graph_class
from cme_python.engines.optimization import jaccard, stem
from cme_python.models import Belief, Change
from cme_python.store import BeliefStore

_WORDS = re.compile(r"[a-z']+")
_NEGATIONS = frozenset(
    [
        "no",
        "not",
        "never",
        "none",
        "cannot",
        "can't",
        "without",
        "nor",
        "lacks",
        "lack",
        "lacking",
        "isn't",
        "doesn't",
        "don't",
        "didn't",
        "won't",
        "wasn't",
        "aren't",
        "hasn't",
        "haven't",
        "unable",
        "fails",
        "fail",
        "false",
        "absent",
        "neither",
    ]
)

CONTRADICTION_AT = 0.75
"""Content overlap above which two oppositely-polarised claims are a clash.

Calibrated, not arbitrary. Around 0.6, "Python has a garbage collector" and
"Rust has no garbage collector" score 0.6 and get flagged — they share every
word but the subject, which is the one word that matters. Raising the bar to
0.75 requires near-identical content, so a disagreement has to be about the
same thing. The cost is that heavily reworded negations slip through; an
entailment model is the real fix, and this is the knob to retune when one lands.
"""

DAMPING = 0.5
"""Each hop away from the source, a confidence shift loses half its force."""

DECISIVE_AT = 0.35
"""Evidence margin above which the losing side is retired, not just weakened.

The margin is in `_weight` units — confidence scaled by the evidence for and
against — so 0.35 is roughly a confident claim with a source facing an
unsupported one. Below it the two sides are close enough that retiring either
would be picking a winner on noise, and the honest act is to trust the weaker
one less and leave both in play.

Deliberately cautious. A wrongly-retired belief leaves recall and stops arguing
its own case; a wrongly-kept one merely stays visible at a lower confidence.
"""


class Contradiction(BaseModel):
    """Two beliefs asserting opposite things about the same subject."""

    a: Belief
    b: Belief
    overlap: float

    @property
    def winner(self) -> Belief:
        """Whichever side the evidence favours."""
        return max((self.a, self.b), key=_weight)

    @property
    def loser(self) -> Belief:
        return self.b if self.winner is self.a else self.a

    def explain(self) -> str:
        return (
            f'"{self.a.statement}" ({self.a.confidence:.0%}) contradicts '
            f'"{self.b.statement}" ({self.b.confidence:.0%}) '
            f"at {self.overlap:.0%} content overlap."
        )


class Resolution(BaseModel):
    """What the engine did about one contradiction, and why."""

    winner: str
    loser: str
    retired: bool
    """True if the loser was superseded; False if it was only weakened."""
    margin: float
    """How far the evidence favoured the winner. Below `DECISIVE_AT`, no retirement."""
    explanation: str


class Chain(BaseModel):
    """A multi-hop reasoning path: belief -> shared concept -> belief -> ..."""

    beliefs: list[Belief]
    concepts: list[str]
    strength: float

    @property
    def hops(self) -> int:
        return max(len(self.beliefs) - 1, 0)

    def explain(self) -> str:
        steps = " -> ".join(
            part
            # strict=False: this is display code, and a malformed chain should
            # render short rather than raise from inside an error message.
            for belief, concept in zip(self.beliefs, [*self.concepts, None], strict=False)
            for part in (f'"{belief.statement}"', concept)
            if part
        )
        return f"{steps}  (strength {self.strength:.2f} over {self.hops} hop(s))"


def _weight(belief: Belief) -> float:
    """How much a belief should count when two of them disagree."""
    support = sum(e.strength for e in belief.evidence if e.supports)
    against = sum(e.strength for e in belief.evidence if not e.supports)
    return belief.confidence * (1 + support - against)


def _polarity(statement: str) -> int:
    """0 for an affirmative claim, 1 for a negated one."""
    return sum(w in _NEGATIONS for w in _WORDS.findall(statement.lower())) % 2


def _content(statement: str) -> set[str]:
    """Stemmed meaning-bearing words, with negation stripped out.

    Negation has to be removed here or it drags the similarity down and the two
    statements stop looking like they are about the same thing — which is
    exactly when we most need to notice that they disagree.
    """
    return {stem(t) for t in tokenise(statement) if t not in _NEGATIONS}


def _clash_score(ca: set[str], cb: set[str], threshold: float) -> float:
    """Overlap of two content sets, or 0.0 if they are not close enough.

    Separate from `contradicts` so the registry-wide scan can reuse content sets
    it has already built, without a second definition of what counts as a clash
    drifting away from this one.
    """
    if not ca or not cb:
        return 0.0
    # Jaccard cannot reach the threshold unless the sets are of comparable size,
    # and that is a length check rather than two set operations.
    if min(len(ca), len(cb)) < threshold * max(len(ca), len(cb)):
        return 0.0
    overlap = len(ca & cb) / len(ca | cb)
    return round(overlap, 6) if overlap >= threshold else 0.0


def contradicts(a: str, b: str, *, threshold: float = CONTRADICTION_AT) -> float:
    """Overlap score if the two statements clash, else 0.0."""
    if _polarity(a) == _polarity(b):
        return 0.0
    return _clash_score(_content(a), _content(b), threshold)


class ReasoningEngine:
    """Multi-hop inference, contradiction detection, and belief propagation."""

    def __init__(self, store: BeliefStore, graph: KnowledgeGraph | None = None) -> None:
        self.store = store
        self._graph = graph
        self._graph_size = -1 if graph is None else len(store)

    @property
    def graph(self) -> KnowledgeGraph:
        """Rebuilt whenever the registry has changed under us.

        Uses the Rust core when it is built, the Python reference otherwise.
        Both produce identical traversals — see `test_graph_parity.py`.
        """
        if self._graph is None or self._graph_size != len(self.store):
            backend = graph_class(settings.graph_backend != "python")
            self._graph = backend.from_store(self.store)
            self._graph_size = len(self.store)
        return self._graph

    # --- multi-hop reasoning ----------------------------------------------

    def connect(self, start_id: str, goal_id: str, *, max_hops: int = 6) -> Chain | None:
        """The reasoning path linking two beliefs, or None if they are unrelated."""
        graph = self.graph
        path = graph.path(belief_node(start_id), belief_node(goal_id), max_hops=max_hops)
        if path is None:
            return None
        beliefs = [graph.belief(n.key) for n in path if n.kind == BELIEF]
        concepts = [n.key for n in path if n.kind == CONCEPT]
        return Chain(
            beliefs=[b for b in beliefs if b is not None],
            concepts=concepts,
            strength=graph.path_strength(path),
        )

    def infer(self, belief_id: str, *, max_hops: int = 4, limit: int = 5) -> list[Chain]:
        """Reasoning chains from one belief out to everything it can reach."""
        chains = []
        for other in self.graph.related(belief_id, max_hops=max_hops):
            chain = self.connect(belief_id, other.id, max_hops=max_hops)
            if chain is not None:
                chains.append(chain)
        chains.sort(key=lambda c: c.strength, reverse=True)
        return chains[:limit]

    # --- contradiction detection ------------------------------------------

    def contradictions(self, *, threshold: float = CONTRADICTION_AT) -> list[Contradiction]:
        """Every pair of stored beliefs that assert opposite things.

        Driven from the negated claims rather than from every pair. A clash
        needs opposite polarity, so only a negated belief can start one, and
        negations are a small minority of anything anyone writes down — the
        affirmative majority is never compared against itself at all.

        Each negated belief then looks up only the beliefs sharing a word with
        it. That is not a heuristic: an overlap of 0.75 cannot be reached by two
        sets with nothing in common, so a candidate that shares no word cannot
        be a clash and the pair never needs scoring.

        ponytail: negation parity is still the signal, and it catches the clash
        that matters — the same claim asserted and denied — with no LLM in the
        loop. An entailment model is the upgrade for reworded disagreements;
        this is only about not paying O(n^2) to find the ones we can already see.
        """
        beliefs = self.store.all()
        content = {b.id: _content(b.statement) for b in beliefs}
        negated = {b.id: _polarity(b.statement) for b in beliefs}

        # Only affirmative claims are indexed: they are the ones a negated
        # belief will be looking for, and indexing the rest would cost memory to
        # find pairs that are thrown away on the polarity check anyway.
        postings: dict[str, list[str]] = defaultdict(list)
        for belief in beliefs:
            if not negated[belief.id]:
                for token in content[belief.id]:
                    postings[token].append(belief.id)

        by_id = {b.id: b for b in beliefs}
        found = []
        for belief in beliefs:
            if not negated[belief.id] or not content[belief.id]:
                continue
            candidates = {
                other for token in content[belief.id] for other in postings.get(token, ())
            }
            for other_id in candidates:
                overlap = _clash_score(content[belief.id], content[other_id], threshold)
                if overlap:
                    found.append(Contradiction(a=by_id[other_id], b=belief, overlap=overlap))
        # Ties broken by id so the order does not depend on set iteration.
        found.sort(key=lambda c: (-c.overlap, c.a.id, c.b.id))
        return found

    def resolve(self, clash: Contradiction, *, force: float = 0.5) -> Belief:
        """Let the better-evidenced side push the other's confidence down.

        Returns the belief that lost ground, saved. Nothing is deleted — a
        belief only disappears once its own confidence decays to zero.
        """
        loser = clash.loser
        loser.confidence = _clamp(loser.confidence * (1 - force * clash.overlap))
        loser.record(Change.CONTRADICTED, clash.winner.id)
        self.store.save(loser)
        return loser

    def reconcile(
        self,
        *,
        threshold: float = CONTRADICTION_AT,
        force: float = 0.5,
        decisive: float = DECISIVE_AT,
    ) -> list[Resolution]:
        """Work through every contradiction in the registry and act on each.

        Detection on its own leaves the registry knowingly inconsistent: the
        engine can already say two beliefs cannot both be true and then serves
        both anyway. This is the other half.

        Two outcomes, because a contradiction is not one situation. Where the
        evidence decisively favours one side, the loser is *superseded* — it
        leaves recall and points at what replaced it. Where the sides are close,
        the loser is only pushed down: a narrow margin is a reason to trust a
        claim less, not a licence to retire it on a rule of thumb.

        A dead heat is left alone. Two claims backed exactly alike give no
        reason to prefer either, and `winner` then falls out of tuple order —
        weakening whichever came first would be dressing a coin toss up as
        inference. It stays in the registry, contradiction and all, for a human
        to break. This is common: one document each, both ingested the same way.

        Re-detecting after each act rather than resolving a snapshot, because
        acting on one clash changes the confidences the next is judged on, and a
        retired belief should stop appearing in later pairs at all. Pairs already
        considered are remembered, so an untouched dead heat cannot be picked up
        forever.
        """
        done: list[Resolution] = []
        seen: set[frozenset[str]] = set()
        while True:
            pending = [
                c
                for c in self.contradictions(threshold=threshold)
                if frozenset((c.a.id, c.b.id)) not in seen
            ]
            if not pending:
                break
            clash = pending[0]
            seen.add(frozenset((clash.a.id, clash.b.id)))

            margin = _weight(clash.winner) - _weight(clash.loser)
            if margin <= 0:
                continue
            if margin >= decisive:
                loser = clash.loser.supersede(clash.winner)
                self.store.save(loser)
            else:
                loser = self.resolve(clash, force=force)
            done.append(
                Resolution(
                    winner=clash.winner.id,
                    loser=loser.id,
                    retired=loser.superseded_by is not None,
                    margin=round(margin, 6),
                    explanation=clash.explain(),
                )
            )
        self._graph_size = -1  # confidences changed; the graph must be rebuilt
        return done

    # --- belief propagation ------------------------------------------------

    def propagate(
        self,
        belief_id: str,
        delta: float,
        *,
        max_hops: int = 2,
        damping: float = DAMPING,
    ) -> dict[str, float]:
        """Shift one belief's confidence and let the change ripple outward.

        Neighbours move in the same direction, weakened by `damping` per hop and
        scaled by how much content they actually share with the source. Returns
        every belief id that moved, mapped to its new confidence.
        """
        source = self.store.get(belief_id)
        if source is None:
            return {}

        source.confidence = _clamp(source.confidence + delta)
        source.record(Change.PROPAGATED, f"{delta:+.3f} applied directly")
        self.store.save(source)
        moved: dict[str, float] = {source.id: source.confidence}

        for node, edges in self.graph.walk(belief_node(belief_id), max_hops=max_hops * 2):
            if node.kind != BELIEF or node.key == belief_id:
                continue
            # Two graph edges (belief -> concept -> belief) is one reasoning hop.
            hops = max(edges // 2, 1)
            if hops > max_hops:
                continue
            neighbour = self.store.get(node.key)
            if neighbour is None:
                continue
            shift = delta * (damping**hops) * jaccard(source.statement, neighbour.statement)
            if abs(shift) < 1e-9:
                continue
            neighbour.confidence = _clamp(neighbour.confidence + shift)
            neighbour.record(Change.PROPAGATED, f"{shift:+.3f} from {source.id}, {hops} hop(s)")
            self.store.save(neighbour)
            moved[neighbour.id] = neighbour.confidence

        self._graph_size = -1  # confidences changed; the graph must be rebuilt
        return moved


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 6)
