"""Qdrant-backed vector index — the store the spec names.

Same surface as the in-memory `VectorIndex`, so `vector_retriever` and the
Evidence Engine cannot tell the difference. The in-memory index stays the
default: it needs no server, and most registries never outgrow it.

`qdrant-client` is imported lazily and is not a CME dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

from cme_python.engines.vectors import DIMENSIONS, Embedder, HashingEmbedder
from cme_python.models import Belief

COLLECTION = "cme_beliefs"


def _missing() -> RuntimeError:
    return RuntimeError(
        "QdrantIndex needs `qdrant-client`, which CME does not install by "
        "default. Run `pip install qdrant-client`, or use the in-memory "
        "VectorIndex."
    )


class QdrantIndex:
    """Vector index backed by Qdrant.

    Drop-in for `VectorIndex`:

        index = QdrantIndex(url="http://localhost:6333")
        engine = EvidenceEngine(store, retriever=vector_retriever(store, index))
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        *,
        embedder: Embedder | None = None,
        collection: str = COLLECTION,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
            from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415
        except ImportError:
            raise _missing() from None

        self.embedder = embedder or HashingEmbedder()
        self.collection = collection
        self._client = QdrantClient(url=url, api_key=api_key)
        self._models = __import__("qdrant_client.models", fromlist=["models"])

        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                # Cosine, matching the in-memory index — a different metric here
                # would silently change what "similar" means between backends.
                vectors_config=VectorParams(
                    size=getattr(self.embedder, "dimensions", DIMENSIONS),
                    distance=Distance.COSINE,
                ),
            )

    def __len__(self) -> int:
        return self._client.count(self.collection, exact=True).count

    def _text(self, belief: Belief) -> str:
        return " ".join([belief.statement, *(e.snippet for e in belief.evidence)])

    def add(self, belief: Belief) -> None:
        self.add_all([belief])

    def add_all(self, beliefs: Sequence[Belief]) -> QdrantIndex:
        if not beliefs:
            return self
        points = [
            self._models.PointStruct(
                # Qdrant ids must be a UUID or an integer; belief ids are hex
                # uuid4 without dashes, so they are reversible either way.
                id=_as_uuid(b.id),
                vector=self.embedder.embed(self._text(b)),
                payload={"belief_id": b.id},
            )
            for b in beliefs
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return self

    def remove(self, belief_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(points=[_as_uuid(belief_id)]),
            wait=True,
        )

    def clear(self) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.FilterSelector(filter=self._models.Filter()),
            wait=True,
        )

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Belief ids nearest the query, best first — same contract as VectorIndex."""
        vector = self.embedder.embed(query)
        if not any(vector):
            return []
        found = self._client.query_points(
            collection_name=self.collection, query=vector, limit=limit, with_payload=True
        ).points
        hits = [(p.payload["belief_id"], round(p.score, 6)) for p in found]
        # Qdrant can return small negative cosines; the in-memory index drops
        # those, so drop them here too or the two backends disagree at the tail.
        hits = [(i, s) for i, s in hits if s > 0]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits


def _as_uuid(belief_id: str) -> str:
    """Belief ids are 32 hex characters; Qdrant wants canonical UUID form."""
    if len(belief_id) != 32:
        return belief_id
    b = belief_id
    return f"{b[:8]}-{b[8:12]}-{b[12:16]}-{b[16:20]}-{b[20:]}"
