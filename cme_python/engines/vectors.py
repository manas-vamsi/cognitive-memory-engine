"""Vector retrieval — the Evidence Engine's semantic backend.

Lexical TF-IDF matches words. It cannot tell that "How do I make an HTTP call?"
and "Sending a web request in Python" are the same question, which is exactly
the failure `ponytail:` notes in `evidence.py` and `optimization.py` have been
pointing at.

Three layers, each replaceable:

    Embedder        text -> vector
    VectorIndex     vectors -> nearest neighbours
    vector_retriever    wires them into the `Retriever` signature

The default `HashingEmbedder` needs no model download, no server and no network,
so semantic-style retrieval works out of the box and tests stay deterministic.
It is a real technique (the hashing trick over character n-grams), not a
placeholder — but it captures surface similarity, not meaning. A sentence
transformer or a hosted embedding endpoint drops in behind `Embedder`, and
Qdrant drops in behind `VectorIndex`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from cme_python.engines.evidence import tokenise
from cme_python.engines.optimization import stem
from cme_python.models import Belief
from cme_python.store import BeliefStore

DIMENSIONS = 1024
"""Measured, not picked for looking round.

Too few dimensions and hash collisions manufacture similarity between unrelated
text. Across the same fixed pairs, the margin between the weakest related score
and the strongest unrelated one is:

    256 dims   -0.065   (overlapping — no threshold can separate them)
    1024 dims  +0.123
    4096 dims  +0.168

256 was the original value and it was actively broken: "cricket scores in 1998"
scored 0.23 against a belief about HTTP, higher than several genuine matches.
4096 separates best but costs 16x the memory per belief for a margin already
comfortable at 1024.

ponytail: 1024 floats per belief is ~8KB in Python, so a 100k-belief registry
is around 800MB in RAM. That is the point to move the index into Qdrant or
pgvector rather than to shrink this number back down.
"""

NGRAM = 4

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    dimensions: int

    def embed(self, text: str) -> Vector: ...


class HashingEmbedder:
    """Hashing trick over word and character n-grams.

    Character n-grams are what make this more than a bag of words: "request"
    and "requests" share almost every 4-gram, so morphology stops being a cliff
    edge the way it is for exact token matching.

    ponytail: deterministic, dependency-free, and honest about its limit — it
    models surface form, not meaning. Swap in a sentence transformer behind the
    `Embedder` protocol when semantic recall actually matters.
    """

    def __init__(self, dimensions: int = DIMENSIONS, ngram: int = NGRAM) -> None:
        self.dimensions = dimensions
        self.ngram = ngram

    def embed(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for feature, weight in self._features(text):
            # Signed hashing: the sign bit keeps unrelated collisions from
            # always accumulating in the same direction.
            slot = hash_feature(feature) % self.dimensions
            sign = 1.0 if hash_feature(feature + "#") % 2 else -1.0
            vector[slot] += sign * weight
        return normalise_vector(vector)

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenise(text)
        features: list[tuple[str, float]] = []
        for word in words:
            features.append((word, 1.0))
            # The stem carries most of the weight of the word itself, so
            # "request" and "requests" share a strong feature instead of
            # relying on n-gram overlap to bridge a one-letter difference.
            root = stem(word)
            if root != word:
                features.append((root, 0.8))
            # No space padding, and only words long enough to have a
            # distinctive interior. Padding manufactures generic n-grams like
            # " in " that collide across unrelated topics — that is how
            # "cricket scores in 1998" came to match a belief about HTTP.
            if len(word) > self.ngram:
                for i in range(len(word) - self.ngram + 1):
                    # Sub-word evidence is weaker than a whole-word match.
                    features.append((word[i : i + self.ngram], 0.35))
        return features


_HASH = re.compile(r"\s+")


def hash_feature(feature: str) -> int:
    """Stable across processes — Python's `hash()` is randomised per run.

    FNV-1a. A retrieval index that reshuffles when the server restarts is not
    an index, so the built-in hash cannot be used here.
    """
    h = 0xCBF29CE484222325
    for byte in feature.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def normalise_vector(vector: Vector) -> Vector:
    length = math.sqrt(sum(v * v for v in vector))
    return vector if length == 0 else [v / length for v in vector]


def cosine(a: Vector, b: Vector) -> float:
    """Both vectors are unit length, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class VectorIndex:
    """In-memory nearest-neighbour search over belief vectors.

    ponytail: brute-force scan. Exact, and fast enough for thousands of
    beliefs; Qdrant or pgvector is the upgrade, and it slots in here without
    the Evidence Engine noticing.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._vectors: dict[str, Vector] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, belief: Belief) -> None:
        # Evidence snippets are indexed with the statement: a belief is
        # findable by what supports it, not only by how it was phrased.
        text = " ".join([belief.statement, *(e.snippet for e in belief.evidence)])
        self._vectors[belief.id] = self.embedder.embed(text)

    def add_all(self, beliefs: Sequence[Belief]) -> VectorIndex:
        for belief in beliefs:
            self.add(belief)
        return self

    def remove(self, belief_id: str) -> None:
        self._vectors.pop(belief_id, None)

    def clear(self) -> None:
        self._vectors.clear()

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Belief ids nearest the query, best first."""
        target = self.embedder.embed(query)
        if not any(target):
            return []
        scored = [
            (belief_id, round(cosine(target, vector), 6))
            for belief_id, vector in self._vectors.items()
        ]
        scored = [(i, s) for i, s in scored if s > 0]
        # Id as the tie-break, so equal scores do not reorder between runs.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


MIN_SIMILARITY = 0.12
"""Below this a match is hash noise, not a topic.

Measured at `DIMENSIONS`: related pairs score 0.17 and up, unrelated pairs 0.045
and below, so this sits in the middle of the gap with room either side. Unlike
lexical overlap, vector similarity is almost never exactly zero, so without a
floor every query returns something — and those beliefs are handed to a model
as "known facts". Wrong facts cost more than missing ones.

The floor applies to raw similarity, before confidence scaling: how well a
belief matches the topic is a separate question from how much we believe it.
"""


def vector_retriever(
    store: BeliefStore,
    index: VectorIndex | None = None,
    *,
    min_similarity: float = MIN_SIMILARITY,
) -> Callable[[str, int], list[tuple[Belief, float]]]:
    """A `Retriever` for `EvidenceEngine`, backed by vector search.

        engine = EvidenceEngine(store, retriever=vector_retriever(store))

    Similarity is scaled by belief confidence for the same reason the lexical
    path does it: a perfectly matching statement nobody believes should not win.

    The index updates incrementally. Rebuilding on every change made a single
    new note cost a full re-embed of the registry — 5.7 seconds at 25k beliefs,
    paid by the next question. For a memory engine, ingesting constantly is the
    normal case, not the edge case.
    """
    index = index or VectorIndex()
    seen: dict[str, str] = {}

    def refresh() -> None:
        current = store.fingerprints()
        if current == seen:
            return
        for gone in seen.keys() - current.keys():
            index.remove(gone)
        # An id whose stamp moved was edited; re-embedding it is an upsert.
        changed = [bid for bid, stamp in current.items() if seen.get(bid) != stamp]
        if changed:
            fresh = [store.get(bid) for bid in changed]
            index.add_all([b for b in fresh if b is not None])
        seen.clear()
        seen.update(current)

    def retrieve(query: str, limit: int) -> list[tuple[Belief, float]]:
        refresh()
        hits = []
        # Over-fetch, because confidence scaling can reorder the shortlist and
        # the best final score may not be the best raw similarity.
        for belief_id, similarity in index.search(query, limit=limit * 3):
            if similarity < min_similarity:
                continue
            belief = store.get(belief_id)
            if belief is None:
                continue
            hits.append((belief, round(similarity * belief.confidence, 6)))
        # Re-sort on the scaled score: ranking by raw similarity here would let
        # a disbelieved statement outrank a well-supported one.
        hits.sort(key=lambda pair: (-pair[1], pair[0].id))
        return hits[:limit]

    return retrieve
