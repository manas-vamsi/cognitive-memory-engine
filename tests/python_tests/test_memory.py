"""Self-check for the Memory Engine. Run: python tests/python_tests/test_memory.py"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.cme import CME
from cme_python.engines.memory import MemoryEngine
from cme_python.models import Belief, MemoryTier
from cme_python.store import BeliefStore

ALICE = "Alice is allergic to penicillin."
BOB = "Bob is allergic to shellfish."
PAPER = "Qubits can hold a superposition of states."


@pytest.fixture
def memory():
    with BeliefStore() as store:
        engine = MemoryEngine(store)
        engine.remember([Belief(statement=ALICE)], tier=MemoryTier.USER, scope="alice")
        engine.remember([Belief(statement=BOB)], tier=MemoryTier.USER, scope="bob")
        engine.remember([Belief(statement=PAPER)], tier=MemoryTier.SCIENTIFIC)
        yield engine


def test_remember_stamps_the_tier_and_scope(memory):
    stored = memory.recall(tier=MemoryTier.USER, scope="alice")
    assert [b.statement for b in stored] == [ALICE]
    assert stored[0].tier == MemoryTier.USER
    assert stored[0].scope == "alice"


def test_recall_is_confined_to_one_tier(memory):
    assert [b.statement for b in memory.recall(tier=MemoryTier.SCIENTIFIC)] == [PAPER]
    assert len(memory.recall(tier=MemoryTier.USER)) == 2
    assert memory.recall(tier=MemoryTier.PROJECT) == []


def test_scope_separates_owners_inside_a_tier(memory):
    """The property that matters: one user's memory must not leak into another's."""
    assert [b.statement for b in memory.recall(tier=MemoryTier.USER, scope="bob")] == [BOB]
    assert ALICE not in [b.statement for b in memory.recall(tier=MemoryTier.USER, scope="bob")]


def test_recall_without_a_tier_sees_everything(memory):
    assert len(memory.recall()) == 3


def test_stats_break_down_by_tier(memory):
    stats = memory.stats()
    assert stats.total == 3
    assert stats.by_tier == {"scientific": 1, "user": 2}


def test_forget_drops_one_scope_and_leaves_the_rest(memory):
    assert memory.forget(tier=MemoryTier.USER, scope="alice") == 1
    assert len(memory.recall()) == 2
    assert [b.statement for b in memory.recall(tier=MemoryTier.USER)] == [BOB]


def test_forget_a_whole_tier(memory):
    assert memory.forget(tier=MemoryTier.USER) == 2
    assert [b.statement for b in memory.recall()] == [PAPER]


def test_prune_removes_disproven_beliefs(memory):
    memory.remember([Belief(statement="Disproven.", confidence=0.0)], tier=MemoryTier.PROJECT)
    assert memory.prune() == 1
    assert len(memory.recall()) == 3


# --- scoped recall through the facade --------------------------------------


@pytest.fixture
def cme():
    with CME(":memory:") as engine:
        engine.ingest(ALICE, tier=MemoryTier.USER, scope="alice")
        engine.ingest(BOB, tier=MemoryTier.USER, scope="bob")
        engine.ingest(PAPER, tier=MemoryTier.SCIENTIFIC)
        yield engine


def test_context_scoped_to_a_user_cannot_see_another(cme):
    ctx = cme.context("allergic", tier=MemoryTier.USER, scope="alice")
    assert [b.statement for b in ctx.beliefs] == [ALICE]


def test_unscoped_context_sees_across_tiers(cme):
    ctx = cme.context("allergic")
    assert {b.statement for b in ctx.beliefs} == {ALICE, BOB}


def test_scoping_applies_before_the_limit(cme):
    """Out-of-scope beliefs must not consume result slots.

    Filtering now runs on cached tier/scope rather than on loaded beliefs, so
    this pins the property that survived the change: fill the registry with
    strongly-matching beliefs owned by someone else, and a scoped query must
    still return the one belief its owner can see — not an empty list because
    the others crowded the shortlist.
    """
    for i in range(20):
        cme.ingest(
            f"Bob is allergic to substance number {i}.",
            tier=MemoryTier.USER,
            scope="bob",
        )
    found = cme.evidence.retrieve(
        "allergic", limit=3, within=cme.memory.view(MemoryTier.USER, "alice")
    )
    assert [b.statement for b, _ in found] == [ALICE]


def test_verification_is_scoped_too(cme):
    """A claim backed only by another scope is not backed for this caller."""
    assert cme.verify(BOB, tier=MemoryTier.USER, scope="bob").is_grounded
    assert not cme.verify(BOB, tier=MemoryTier.USER, scope="alice").is_grounded


def test_stats_through_the_facade(cme):
    assert cme.stats().by_tier == {"scientific": 1, "user": 2}


# --- migration --------------------------------------------------------------


def test_a_database_written_before_tiers_still_opens(tmp_path):
    """An older registry is somebody's accumulated memory; it must not crash."""
    db = tmp_path / "legacy.sqlite"
    legacy = sqlite3.connect(db)
    legacy.execute(
        """CREATE TABLE beliefs (
               id TEXT PRIMARY KEY, statement TEXT NOT NULL, confidence REAL NOT NULL,
               source TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
               data TEXT NOT NULL)"""
    )
    old = Belief(statement="Written before tiers existed.", confidence=0.8)
    legacy.execute(
        "INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            old.id,
            old.statement,
            old.confidence,
            str(old.source),
            old.created_at.isoformat(),
            old.updated_at.isoformat(),
            old.model_dump_json(),
        ),
    )
    legacy.commit()
    legacy.close()

    with BeliefStore(db) as store:
        recovered = store.get(old.id)
        assert recovered is not None
        assert recovered.statement == old.statement
        assert recovered.tier == MemoryTier.GENERAL
        assert store.count_by_tier() == {"general": 1}
        assert store.all(tier=MemoryTier.GENERAL) != []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
