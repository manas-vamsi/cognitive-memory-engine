"""Self-check for the belief registry. Run: python tests/python_tests/test_store.py"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.models import DEAD_BELOW, Belief, Evidence, SourceKind
from cme_python.store import BeliefStore

DATABASE_URL = os.environ.get("DATABASE_URL", "")

BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param(
        "postgres",
        id="postgres",
        marks=pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set"),
    ),
]


@pytest.fixture(params=BACKENDS)
def store(request):
    """Every test below runs against both backends.

    The registry contract is the same either way; testing Postgres with its own
    parallel suite would let the two drift, which is exactly the bug worth
    preventing.
    """
    if request.param == "sqlite":
        with BeliefStore() as s:
            yield s
        return

    from cme_python.store import open_store

    s = open_store(DATABASE_URL)
    s._exec("DELETE FROM beliefs")  # a shared server needs a clean table
    try:
        yield s
    finally:
        s._exec("DELETE FROM beliefs")
        s.close()


def test_round_trip_preserves_evidence_and_connections(store):
    b = Belief(statement="Qubits can be entangled.", source=SourceKind.RESEARCH_PAPER)
    b.add_evidence(Evidence(snippet="Bell test", strength=0.9, locator="doi:10/x"))
    b.connect("Quantum Computing", "Entanglement")
    store.save(b)

    got = store.get(b.id)
    assert got is not None
    assert got.statement == b.statement
    assert got.confidence == b.confidence
    assert got.connections == {"Quantum Computing", "Entanglement"}
    assert [e.locator for e in got.evidence] == ["doi:10/x"]


def test_save_is_idempotent_and_updates_in_place(store):
    b = store.save(Belief(statement="Python is used in ML."))
    b.add_evidence(Evidence(snippet="PyTorch docs", strength=0.8))
    store.save(b)
    assert len(store) == 1
    assert store.get(b.id).confidence > 0.5


def test_missing_and_deleted_beliefs(store):
    assert store.get("nope") is None
    b = store.save(Belief(statement="Temporary."))
    assert store.delete(b.id) is True
    assert store.delete(b.id) is False
    assert len(store) == 0


def test_all_filters_by_confidence_and_sorts_strongest_first(store):
    store.save_all(
        [
            Belief(statement="weak", confidence=0.2),
            Belief(statement="strong", confidence=0.95),
            Belief(statement="middling", confidence=0.6),
        ]
    )
    assert [b.statement for b in store.all()] == ["strong", "middling", "weak"]
    assert [b.statement for b in store.all(min_confidence=0.5)] == ["strong", "middling"]
    assert len(store.all(limit=1)) == 1


def test_search_matches_statement_substring(store):
    store.save_all([Belief(statement="Rust is fast."), Belief(statement="Python is slow.")])
    assert [b.statement for b in store.search("Rust")] == ["Rust is fast."]
    assert store.search("Haskell") == []


def test_search_ignores_case_on_every_backend(store):
    """A dialect difference with teeth.

    SQLite's LIKE is case-insensitive and Postgres's is not. Duplicate
    detection in the Belief Engine probes through `search`, so a case-sensitive
    backend would quietly stop recognising "Rust is fast" and "rust is fast" as
    the same claim — the same input building a different registry depending on
    where it is stored.
    """
    store.save(Belief(statement="Rust Is Fast."))
    assert [b.statement for b in store.search("rust is fast")] == ["Rust Is Fast."]
    assert [b.statement for b in store.search("RUST")] == ["Rust Is Fast."]


def test_prune_removes_only_disproven_beliefs(store):
    store.save_all(
        [Belief(statement="disproven", confidence=0.0), Belief(statement="alive", confidence=0.4)]
    )
    assert store.prune_dead(DEAD_BELOW) == 1
    assert [b.statement for b in store.all()] == ["alive"]


def test_usable_from_more_than_one_thread(store):
    """A threadpooled web server touches the connection from many threads."""
    import threading

    errors: list[Exception] = []

    def work(n: int) -> None:
        try:
            store.save(Belief(statement=f"Claim number {n} from a worker thread."))
            store.all()
            len(store)
        except Exception as exc:  # noqa: BLE001 - the point is to surface it
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store) == 8


def test_survives_reopening_the_file(tmp_path):
    db = tmp_path / "nested" / "cme.sqlite"
    with BeliefStore(db) as s:
        bid = s.save(Belief(statement="Persisted across sessions.")).id
    with BeliefStore(db) as s:
        assert s.get(bid).statement == "Persisted across sessions."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
