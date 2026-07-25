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

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
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

Raising it costs less than it looks: vectors are stored sparsely, so a belief
occupies its ~28 non-zero dimensions rather than all 1024. Measured at ~5.5KB
per belief including the postings, so a 100k-belief registry is roughly 550MB —
the point to move the index into Qdrant rather than to shrink this number.
"""

NGRAM = 4

Vector = dict[int, float]
"""Sparse: dimension index -> weight, with zeros absent.

A belief's embedding touches maybe sixty of `DIMENSIONS` slots, so a dense list
is ~94% zeros. Storing them meant every comparison multiplied those zeros
together — the cost of a query scaled with the size of the vector space rather
than with the content of the two texts.
"""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a sparse vector."""

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
        vector: Vector = {}
        for feature, weight in self._features(text):
            # Signed hashing: the sign bit keeps unrelated collisions from
            # always accumulating in the same direction.
            slot = hash_feature(feature) % self.dimensions
            sign = 1.0 if hash_feature(feature + "#") % 2 else -1.0
            vector[slot] = vector.get(slot, 0.0) + sign * weight
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
    length = math.sqrt(sum(v * v for v in vector.values()))
    return vector if length == 0 else {i: v / length for i, v in vector.items()}


def cosine(a: Vector, b: Vector) -> float:
    """Both vectors are unit length, so the dot product is the cosine.

    Walks the shorter vector: only shared dimensions contribute, and a missing
    dimension is a zero that need not be visited.
    """
    if len(b) < len(a):
        a, b = b, a
    return sum(weight * b[i] for i, weight in a.items() if i in b)


def to_dense(vector: Vector, dimensions: int = DIMENSIONS) -> list[float]:
    """Expand to a plain list, for stores that expect a dense vector."""
    dense = [0.0] * dimensions
    for index, weight in vector.items():
        dense[index] = weight
    return dense


class VectorIndex:
    """In-memory nearest-neighbour search over belief vectors.

    Exact, and now proportional to how many beliefs share a dimension with the
    query rather than to how many exist. Qdrant is still the upgrade for large
    registries, and slots in here without the Evidence Engine noticing.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._vectors: dict[str, Vector] = {}
        self._dims: dict[int, set[str]] = defaultdict(set)
        """dimension -> beliefs with a non-zero weight there.

        The same idea as the lexical postings list. A query touches sixty-odd
        dimensions; without this, every query scored every belief in the
        registry, which is why 10k beliefs cost ~600ms.
        """

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, belief: Belief) -> None:
        # Evidence snippets are indexed with the statement: a belief is
        # findable by what supports it, not only by how it was phrased.
        text = " ".join([belief.statement, *(e.snippet for e in belief.evidence)])
        self._unindex(belief.id)
        vector = self.embedder.embed(text)
        self._vectors[belief.id] = vector
        for index in vector:
            self._dims[index].add(belief.id)

    def add_all(self, beliefs: Sequence[Belief]) -> VectorIndex:
        for belief in beliefs:
            self.add(belief)
        return self

    def _unindex(self, belief_id: str) -> None:
        for index in self._vectors.get(belief_id, {}):
            postings = self._dims.get(index)
            if postings is None:
                continue
            postings.discard(belief_id)
            if not postings:
                del self._dims[index]

    def remove(self, belief_id: str) -> None:
        self._unindex(belief_id)
        self._vectors.pop(belief_id, None)

    def clear(self) -> None:
        self._vectors.clear()
        self._dims.clear()

    @property
    def vectors(self) -> dict[str, Vector]:
        """Read-only view for persistence."""
        return self._vectors

    def load(self, vectors: dict[str, Vector]) -> None:
        """Replace the contents with already-computed vectors."""
        self.clear()
        for belief_id, vector in vectors.items():
            self._vectors[belief_id] = vector
            for index in vector:
                self._dims[index].add(belief_id)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Belief ids nearest the query, best first."""
        target = self.embedder.embed(query)
        if not target:
            return []
        # Only beliefs sharing a dimension can have a non-zero dot product.
        candidates: set[str] = set()
        for index in target:
            candidates |= self._dims.get(index, frozenset())
        scored = [(bid, round(cosine(target, self._vectors[bid]), 6)) for bid in candidates]
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


CACHE_VERSION = 1


class VectorRetriever:
    """A `Retriever` for `EvidenceEngine`, backed by vector search.

        engine = EvidenceEngine(store, retriever=VectorRetriever(store))

    Similarity is scaled by belief confidence for the same reason the lexical
    path does it: a perfectly matching statement nobody believes should not win.

    The index updates incrementally. Rebuilding on every change made a single
    new note cost a full re-embed of the registry — 5.7 seconds at 25k beliefs,
    paid by the next question. For a memory engine, ingesting constantly is the
    normal case, not the edge case.
    """

    def __init__(
        self,
        store: BeliefStore,
        index: VectorIndex | None = None,
        *,
        min_similarity: float = MIN_SIMILARITY,
        cache: str | Path | None = None,
    ) -> None:
        self.store = store
        self.index = index or VectorIndex()
        self.min_similarity = min_similarity
        # Embeddings are the expensive part of a cold start — 618ms of the
        # 765ms to index 10k beliefs — and they do not change unless the belief
        # does. A per-registry cache turns process start into a load.
        self.cache = Path(cache) if cache else None
        self.seen: dict[str, str] = {}
        self._loaded = False
        self._dirty = False

    # --- persistence -------------------------------------------------------

    def _signature(self) -> dict[str, object]:
        embedder = self.index.embedder
        return {
            "version": CACHE_VERSION,
            "dimensions": getattr(embedder, "dimensions", DIMENSIONS),
            "ngram": getattr(embedder, "ngram", NGRAM),
            "embedder": type(embedder).__name__,
        }

    def load(self) -> None:
        """Populate the index from the cache, if one matches this embedder.

        The signature check is not paperwork: vectors written with different
        dimensions or a different embedder are not wrong-looking, they are
        silently meaningless, and every similarity computed against them would
        be nonsense that still ranks. A mismatch discards the cache.
        """
        self._loaded = True
        if self.cache is None or not self.cache.exists():
            return
        try:
            payload = json.loads(self.cache.read_text(encoding="utf-8"))
            if payload.get("signature") != self._signature():
                return
            vectors = {
                belief_id: {int(dim): weight for dim, weight in vector.items()}
                for belief_id, vector in payload["vectors"].items()
            }
            fingerprints = payload["fingerprints"]
        except (OSError, ValueError, KeyError, AttributeError):
            # A corrupt or half-written cache is a performance problem, never a
            # correctness one: drop it and rebuild.
            return
        self.index.load(vectors)
        self.seen = dict(fingerprints)

    def persist(self) -> None:
        """Write the index out. Called at shutdown, not on every change."""
        if self.cache is None or not self._dirty:
            # Rewriting an unchanged cache costs ~750ms at 10k beliefs and buys
            # a byte-identical file. A read-only session should shut down clean.
            return
        payload = {
            "signature": self._signature(),
            "fingerprints": self.seen,
            "vectors": {
                belief_id: {str(dim): round(weight, 6) for dim, weight in vector.items()}
                for belief_id, vector in self.index.vectors.items()
            },
        }
        try:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            # Write beside and rename, so a crash mid-write cannot leave a
            # truncated cache that the next start has to detect.
            temporary = self.cache.with_suffix(self.cache.suffix + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.cache)
            self._dirty = False
        except OSError:
            return

    # --- retrieval ---------------------------------------------------------

    def refresh(self) -> None:
        if not self._loaded:
            self.load()
        store, index, seen = self.store, self.index, self.seen
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
        self.seen = current
        self._dirty = True

    def __call__(self, query: str, limit: int) -> list[tuple[Belief, float]]:
        self.refresh()
        store, index, min_similarity = self.store, self.index, self.min_similarity
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


def vector_retriever(
    store: BeliefStore,
    index: VectorIndex | None = None,
    *,
    min_similarity: float = MIN_SIMILARITY,
    cache: str | Path | None = None,
) -> VectorRetriever:
    """Build a `VectorRetriever`. Kept as a function for existing callers."""
    return VectorRetriever(store, index, min_similarity=min_similarity, cache=cache)


def cache_path_for(database: str) -> Path | None:
    """Where a registry's vector cache lives, or None if it has nowhere to live.

    An in-memory registry has no identity to key a cache to, and caching it
    would mean one run reading another run's vectors for beliefs that no longer
    exist.
    """
    if not database or database == ":memory:" or "://" in database:
        return None
    return Path(database).with_suffix(Path(database).suffix + ".vectors.json")
