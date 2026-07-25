"""Rust-backed knowledge graph — Phase 3 acceleration.

Same interface as the Python `KnowledgeGraph`, with traversal delegated to the
`cme_core` extension. Beliefs and concept labels stay on the Python side; only
adjacency and BFS cross the boundary, because that is the part that costs.

The Python implementation remains the reference. `test_graph_parity.py` runs the
same assertions against both, so "faster" can never quietly become "different".
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from cme_python.engines.belief import normalise
from cme_python.engines.graph import BELIEF, CONCEPT, KnowledgeGraph, Node
from cme_python.models import Belief
from cme_python.store import BeliefStore

try:  # pragma: no cover - import guard, exercised by whether the wheel is built
    import cme_core

    AVAILABLE = True
except ImportError:  # pragma: no cover
    cme_core = None
    AVAILABLE = False


class NativeKnowledgeGraph:
    """Bipartite belief/concept graph backed by `cme_core`."""

    def __init__(self) -> None:
        if not AVAILABLE:
            raise RuntimeError(
                "cme_core is not built. Run `maturin build --release "
                "--features extension-module` in cme-core/, or use KnowledgeGraph."
            )
        self._graph = cme_core.BeliefGraph()
        self._beliefs: dict[str, Belief] = {}
        self._labels: dict[str, str] = {}

    @classmethod
    def from_store(cls, store: BeliefStore, *, min_confidence: float = 0.0) -> NativeKnowledgeGraph:
        return cls().add_all(store.all(min_confidence=min_confidence))

    # --- construction ------------------------------------------------------

    def add(self, belief: Belief) -> NativeKnowledgeGraph:
        self._beliefs[belief.id] = belief
        keys = []
        for label in belief.connections:
            key = normalise(label)
            if not key:
                continue
            self._labels.setdefault(key, label)
            keys.append(key)
        self._graph.add_belief(belief.id, belief.confidence, keys)
        return self

    def add_all(self, beliefs: Iterable[Belief]) -> NativeKnowledgeGraph:
        for belief in beliefs:
            self.add(belief)
        return self

    # --- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, node: Node) -> bool:
        if node.kind == BELIEF:
            return node.key in self._beliefs
        return node.key in self._labels

    @property
    def concepts(self) -> list[str]:
        return sorted(self._labels.values())

    def belief(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def neighbours(self, node: Node) -> set[Node]:
        keys = self._graph.neighbours(node.kind, node.key)
        # A belief node's neighbours are concepts and vice versa — the graph is
        # bipartite, so the kind flips on every edge.
        other = CONCEPT if node.kind == BELIEF else BELIEF
        return {Node(other, key) for key in keys}

    def beliefs_about(self, concept: str) -> list[Belief]:
        keys = self._graph.neighbours(CONCEPT, normalise(concept))
        found = (self._beliefs[k] for k in keys if k in self._beliefs)
        return sorted(found, key=lambda b: b.confidence, reverse=True)

    # --- traversal ---------------------------------------------------------

    def walk(self, start: Node, *, max_hops: int = 3) -> Iterator[tuple[Node, int]]:
        for kind, key, depth in self._graph.walk(start.kind, start.key, max_hops):
            yield Node(kind, key), depth

    def related(self, belief_id: str, *, max_hops: int = 2) -> list[Belief]:
        ids = self._graph.related(belief_id, max_hops)
        return [self._beliefs[i] for i in ids if i in self._beliefs]

    def path(self, start: Node, goal: Node, *, max_hops: int = 6) -> list[Node] | None:
        chain = self._graph.path(start.kind, start.key, goal.kind, goal.key, max_hops)
        return None if chain is None else [Node(kind, key) for kind, key in chain]

    def path_strength(self, path: list[Node]) -> float:
        return round(self._graph.path_strength([(n.kind, n.key) for n in path]), 6)


def graph_class(prefer_native: bool = True) -> type:
    """The graph implementation to use: native when built, Python otherwise."""
    return NativeKnowledgeGraph if (prefer_native and AVAILABLE) else KnowledgeGraph
