"""Qdrant index tests — skipped unless a server is reachable.

Run against a real Qdrant:

    docker run -p 6333:6333 qdrant/qdrant
    QDRANT_URL=http://localhost:6333 pytest tests/python_tests/test_qdrant.py

CI runs these with a Qdrant service container, so the backend is exercised
against the real thing rather than a mock. A mock here would only prove the
mock agrees with itself.
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.evidence import EvidenceEngine
from cme_python.engines.vectors import VectorIndex, vector_retriever
from cme_python.models import Belief
from cme_python.store import BeliefStore

QDRANT_URL = os.environ.get("QDRANT_URL", "")

pytestmark = pytest.mark.skipif(not QDRANT_URL, reason="QDRANT_URL not set")

HTTP = "Sending a web request in Python uses the requests library."
QUBITS = "Qubits can hold a superposition of states."


@pytest.fixture
def index():
    from cme_python.engines.qdrant_index import QdrantIndex

    # A collection per test run, so a failure cannot poison the next one.
    idx = QdrantIndex(QDRANT_URL, collection=f"cme_test_{uuid.uuid4().hex[:8]}")
    yield idx
    idx._client.delete_collection(idx.collection)


def test_add_and_search_finds_the_matching_belief(index):
    http, qubits = Belief(statement=HTTP), Belief(statement=QUBITS)
    index.add_all([http, qubits])
    assert len(index) == 2
    assert index.search("requests library web")[0][0] == http.id


def test_unrelated_query_scores_below_the_floor(index):
    from cme_python.engines.vectors import MIN_SIMILARITY

    index.add_all([Belief(statement=QUBITS)])
    hits = index.search("medieval falconry")
    assert all(score < MIN_SIMILARITY for _, score in hits)


def test_upsert_is_idempotent(index):
    belief = Belief(statement=HTTP)
    index.add(belief)
    index.add(belief)
    assert len(index) == 1


def test_remove_and_clear(index):
    http, qubits = Belief(statement=HTTP), Belief(statement=QUBITS)
    index.add_all([http, qubits])
    index.remove(http.id)
    assert len(index) == 1
    index.clear()
    assert len(index) == 0


def test_empty_query_returns_nothing(index):
    index.add_all([Belief(statement=HTTP)])
    assert index.search("") == []


def test_it_agrees_with_the_in_memory_index(index):
    """The contract that makes it a drop-in: same ranking, same scores."""
    beliefs = [
        Belief(statement=HTTP),
        Belief(statement=QUBITS),
        Belief(statement="Rust guarantees memory safety without a garbage collector."),
    ]
    index.add_all(beliefs)
    memory = VectorIndex().add_all(beliefs)

    for query in ("requests library", "qubit superposition", "memory safety rust"):
        remote = index.search(query, limit=3)
        local = memory.search(query, limit=3)
        assert [i for i, _ in remote] == [i for i, _ in local], query
        for (_, a), (_, b) in zip(remote, local, strict=True):
            assert a == pytest.approx(b, abs=1e-4), query


def test_it_works_as_the_evidence_engine_backend(index):
    with BeliefStore() as store:
        store.save_all(
            [Belief(statement=HTTP, confidence=0.9), Belief(statement=QUBITS, confidence=0.9)]
        )
        evidence = EvidenceEngine(store, retriever=vector_retriever(store, index))
        hits = evidence.retrieve("requesting a webpage")
        assert hits and hits[0][0].statement == HTTP
        assert evidence.retrieve("medieval falconry") == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
