"""Knowledge Graph — the reasoning substrate.

Beliefs and the concepts they mention form one bipartite graph: a belief links
to every concept it is connected to, and two beliefs become reachable through
any concept they share. Multi-hop reasoning is a walk over this graph.

ponytail: dict-of-sets adjacency, in-memory, stdlib only. This is the Python
reference implementation the Rust `cme_core` (petgraph) replaces in Phase 3 —
the interface below is what the wrapper has to keep.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from cme_python.engines.belief import normalise
from cme_python.models import Belief
from cme_python.store import BeliefStore

BELIEF = "belief"
CONCEPT = "concept"


class Node(NamedTuple):
    kind: str
    key: str

    def __repr__(self) -> str:  # keeps traversal output readable
        return f"{self.kind}:{self.key}"


class KnowledgeGraph:
    """Bipartite belief/concept graph with BFS traversal."""

    def __init__(self) -> None:
        self._edges: dict[Node, set[Node]] = {}
        self._beliefs: dict[str, Belief] = {}
        self._labels: dict[str, str] = {}
        """Normalised concept key -> the label as first written, for display."""

    # --- construction ------------------------------------------------------

    @classmethod
    def from_store(cls, store: BeliefStore, *, min_confidence: float = 0.0) -> KnowledgeGraph:
        """Build a graph from the registry, skipping beliefs below the floor."""
        g = cls()
        g.add_all(store.all(min_confidence=min_confidence))
        return g

    def add(self, belief: Belief) -> KnowledgeGraph:
        node = Node(BELIEF, belief.id)
        self._beliefs[belief.id] = belief
        self._edges.setdefault(node, set())
        for label in belief.connections:
            key = normalise(label)
            if not key:
                continue
            self._labels.setdefault(key, label)
            self._link(node, Node(CONCEPT, key))
        return self

    def add_all(self, beliefs: Iterable[Belief]) -> KnowledgeGraph:
        for b in beliefs:
            self.add(b)
        return self

    def _link(self, a: Node, b: Node) -> None:
        self._edges.setdefault(a, set()).add(b)
        self._edges.setdefault(b, set()).add(a)

    # --- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._edges)

    def __contains__(self, node: Node) -> bool:
        return node in self._edges

    @property
    def concepts(self) -> list[str]:
        """Concept labels as originally written, alphabetical."""
        return sorted(self._labels.values())

    def belief(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def neighbours(self, node: Node) -> set[Node]:
        return set(self._edges.get(node, ()))

    def beliefs_about(self, concept: str) -> list[Belief]:
        """Beliefs mentioning a concept, strongest first."""
        node = Node(CONCEPT, normalise(concept))
        found = (self._beliefs[n.key] for n in self._edges.get(node, ()) if n.kind == BELIEF)
        return sorted(found, key=lambda b: b.confidence, reverse=True)

    # --- traversal ---------------------------------------------------------

    def walk(self, start: Node, *, max_hops: int = 3) -> Iterator[tuple[Node, int]]:
        """Breadth-first walk outward from a node, yielding (node, hop distance)."""
        if start not in self._edges:
            return
        seen = {start}
        queue: deque[tuple[Node, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            yield node, depth
            if depth >= max_hops:
                continue
            for nxt in self._edges[node] - seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))

    def related(self, belief_id: str, *, max_hops: int = 2) -> list[Belief]:
        """Beliefs reachable within N hops, nearest first then strongest.

        One hop lands on a shared concept, two hops on the beliefs that also
        mention it — so `max_hops=2` is "other beliefs about the same things".
        """
        out: list[tuple[int, float, Belief]] = []
        for node, depth in self.walk(Node(BELIEF, belief_id), max_hops=max_hops):
            if node.kind == BELIEF and node.key != belief_id:
                b = self._beliefs[node.key]
                out.append((depth, -b.confidence, b))
        out.sort(key=lambda t: (t[0], t[1]))
        return [b for _, _, b in out]

    def path(self, start: Node, goal: Node, *, max_hops: int = 6) -> list[Node] | None:
        """Shortest chain of nodes connecting two points, or None if unreachable.

        This is the multi-hop reasoning trace: A -> shared concept -> B.
        """
        if start not in self._edges or goal not in self._edges:
            return None
        if start == goal:
            return [start]
        came_from: dict[Node, Node] = {start: start}
        queue: deque[tuple[Node, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for nxt in self._edges[node]:
                if nxt in came_from:
                    continue
                came_from[nxt] = node
                if nxt == goal:
                    return _trace(came_from, start, goal)
                queue.append((nxt, depth + 1))
        return None

    def path_strength(self, path: list[Node]) -> float:
        """Confidence of a reasoning chain: the product of the beliefs it crosses.

        A chain is only as trustworthy as its weakest link, and every extra hop
        costs — which is exactly what the Optimization Engine will be minimising.
        """
        strength = 1.0
        for node in path:
            if node.kind == BELIEF:
                strength *= self._beliefs[node.key].confidence
        return round(strength, 6)


def _trace(came_from: dict[Node, Node], start: Node, goal: Node) -> list[Node]:
    chain = [goal]
    while chain[-1] != start:
        chain.append(came_from[chain[-1]])
    return list(reversed(chain))


def belief_node(belief: Belief | str) -> Node:
    return Node(BELIEF, belief if isinstance(belief, str) else belief.id)


def concept_node(label: str) -> Node:
    return Node(CONCEPT, normalise(label))
