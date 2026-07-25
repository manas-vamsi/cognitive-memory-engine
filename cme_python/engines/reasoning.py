"""Reasoning Engine — inference over the belief graph.

Three jobs from the brief: multi-hop reasoning (A → B → C), contradiction
detection (flagging beliefs that clash), and belief propagation (when one
belief's confidence moves, everything resting on it moves too).
"""

from __future__ import annotations

import re
from itertools import combinations

from pydantic import BaseModel

from cme_python.engines.evidence import tokenise
from cme_python.engines.graph import BELIEF, CONCEPT, KnowledgeGraph, belief_node
from cme_python.engines.optimization import jaccard, stem
from cme_python.models import Belief
from cme_python.store import BeliefStore

_WORDS = re.compile(r"[a-z']+")
_NEGATIONS = frozenset(
    """no not never none cannot can't without nor lacks lack lacking
    isn't doesn't don't didn't won't wasn't aren't hasn't haven't unable
    fails fail false absent neither""".split()
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
            for belief, concept in zip(self.beliefs, [*self.concepts, None])
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


def contradicts(a: str, b: str, *, threshold: float = CONTRADICTION_AT) -> float:
    """Overlap score if the two statements clash, else 0.0."""
    if _polarity(a) == _polarity(b):
        return 0.0
    ca, cb = _content(a), _content(b)
    if not ca or not cb:
        return 0.0
    overlap = len(ca & cb) / len(ca | cb)
    return round(overlap, 6) if overlap >= threshold else 0.0


class ReasoningEngine:
    """Multi-hop inference, contradiction detection, and belief propagation."""

    def __init__(self, store: BeliefStore, graph: KnowledgeGraph | None = None) -> None:
        self.store = store
        self._graph = graph
        self._graph_size = -1 if graph is None else len(store)

    @property
    def graph(self) -> KnowledgeGraph:
        """Rebuilt whenever the registry has changed under us."""
        if self._graph is None or self._graph_size != len(self.store):
            self._graph = KnowledgeGraph.from_store(self.store)
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

        ponytail: full pairwise scan with negation parity as the signal. It
        catches the clash that matters — the same claim asserted and denied —
        with no LLM in the loop. Blocking on shared concepts, or an entailment
        model, is the upgrade when the registry outgrows an O(n^2) pass.
        """
        beliefs = self.store.all()
        found = []
        for a, b in combinations(beliefs, 2):
            overlap = contradicts(a.statement, b.statement, threshold=threshold)
            if overlap:
                found.append(Contradiction(a=a, b=b, overlap=overlap))
        found.sort(key=lambda c: c.overlap, reverse=True)
        return found

    def resolve(self, clash: Contradiction, *, force: float = 0.5) -> Belief:
        """Let the better-evidenced side push the other's confidence down.

        Returns the belief that lost ground, saved. Nothing is deleted — a
        belief only disappears once its own confidence decays to zero.
        """
        loser = clash.loser
        loser.confidence = _clamp(loser.confidence * (1 - force * clash.overlap))
        self.store.save(loser)
        return loser

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
            self.store.save(neighbour)
            moved[neighbour.id] = neighbour.confidence

        self._graph_size = -1  # confidences changed; the graph must be rebuilt
        return moved


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 6)
