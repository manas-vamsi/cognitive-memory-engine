"""Self-check for the Memory Engine. Run: python tests/python_tests/test_memory.py"""

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.cme import CME
from cme_python.engines.memory import DECAY_FLOOR, MemoryEngine
from cme_python.models import Belief, Change, Evidence, MemoryTier
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


# --- decay ------------------------------------------------------------------


def _aged(statement: str, days: float, confidence: float = 0.8) -> Belief:
    belief = Belief(statement=statement, confidence=confidence)
    belief.confidence_at = datetime.now(UTC) - timedelta(days=days)
    return belief


def test_an_untouched_belief_halves_over_a_half_life(memory):
    old = memory.remember([_aged("Nobody has mentioned this in six months.", 180)])[0]
    moved = memory.decay(half_life_days=180)
    assert moved[old.id] == pytest.approx(0.4, abs=0.01)


def test_a_fresh_belief_is_untouched(memory):
    fresh = memory.remember([Belief(statement="Recorded just now.", confidence=0.8)])[0]
    assert fresh.id not in memory.decay(half_life_days=180)


def test_decay_never_disproves_a_belief_on_age_alone(memory):
    """Silence is not evidence. A decade of neglect must not reach zero.

    Reaching zero would let `prune` delete it, which would turn "nobody
    mentioned it" into "it was refuted".
    """
    ancient = memory.remember([_aged("True but unfashionable.", 3650)])[0]
    assert memory.decay(half_life_days=180)[ancient.id] == DECAY_FLOOR
    assert memory.prune() == 0


def test_reinforcement_resets_the_clock(memory):
    old = memory.remember([_aged("Stale until someone backs it.", 180)])[0]
    old.add_evidence(Evidence(snippet="Confirmed again today", strength=0.6))
    memory.store.save(old)
    assert old.id not in memory.decay(half_life_days=180)


def test_decay_does_not_compound(memory):
    """Two runs a moment apart must land where one run did.

    If each call halved from the belief's whole age again, a busy scheduler
    would erase a registry in an afternoon.
    """
    old = memory.remember([_aged("Untouched for a year.", 365)])[0]
    first = memory.decay(half_life_days=180)[old.id]
    second = memory.decay(half_life_days=180)
    assert second.get(old.id, first) == pytest.approx(first, abs=1e-6)


def test_decay_is_the_same_in_one_step_or_many(memory):
    """Halving is memoryless, so cadence must not change the destination."""
    now = datetime.now(UTC)
    at_once, in_stages = (
        memory.remember([_aged(f"Aged {how}.", 360)], tier=MemoryTier.USER, scope=how)[0]
        for how in ("at-once", "in-stages")
    )
    memory.decay(half_life_days=180, now=now, tier=MemoryTier.USER, scope="at-once")
    for days_ago in (270, 180, 90, 0):
        memory.decay(
            half_life_days=180,
            now=now - timedelta(days=days_ago),
            tier=MemoryTier.USER,
            scope="in-stages",
        )
    assert memory.store.get(in_stages.id).confidence == pytest.approx(
        memory.store.get(at_once.id).confidence, abs=1e-5
    )


def test_a_legacy_row_ages_from_its_last_write(memory):
    """Rows predating the stamp must not all read as freshly calculated."""
    raw = Belief(statement="Written before decay existed.", confidence=0.8).model_dump(mode="json")
    raw["updated_at"] = (datetime.now(UTC) - timedelta(days=180)).isoformat()
    del raw["confidence_at"]
    memory.remember([Belief.model_validate(raw)])
    assert list(memory.decay(half_life_days=180).values()) == [pytest.approx(0.4, abs=0.01)]


def test_decay_can_be_confined_to_one_slice(memory):
    mine = memory.remember([_aged("Alice's old note.", 180)], tier=MemoryTier.USER, scope="alice")
    theirs = memory.remember([_aged("Bob's old note.", 180)], tier=MemoryTier.USER, scope="bob")
    moved = memory.decay(half_life_days=180, tier=MemoryTier.USER, scope="alice")
    assert mine[0].id in moved
    assert theirs[0].id not in moved


def test_decayed_confidence_reaches_retrieval(memory):
    """The stamp split has to leave indexes able to see the change."""
    from cme_python.engines.evidence import EvidenceEngine

    old = memory.remember([_aged("Quantum entanglement correlates two qubits.", 720)])[0]
    engine = EvidenceEngine(memory.store)
    before = engine.retrieve("entanglement qubits")[0][1]
    memory.decay(half_life_days=180)
    after = engine.retrieve("entanglement qubits")[0][1]
    assert after < before
    assert memory.store.get(old.id).confidence < 0.8


# --- supersession and the timeline ------------------------------------------


def test_a_superseded_belief_leaves_recall_but_not_the_registry(memory):
    old, new = memory.remember(
        [Belief(statement="The rate is 4%."), Belief(statement="The rate is 5%.")],
        tier=MemoryTier.PROJECT,
    )
    memory.supersede(old.id, new.id)
    assert [b.statement for b in memory.recall(tier=MemoryTier.PROJECT)] == ["The rate is 5%."]
    assert memory.store.get(old.id).superseded_by == new.id


def test_deleting_a_tier_takes_the_retired_beliefs_too(memory):
    """Someone exercising deletion means all of it, retired or not."""
    old, new = memory.remember(
        [Belief(statement="Old rate."), Belief(statement="New rate.")],
        tier=MemoryTier.PROJECT,
        scope="rates",
    )
    memory.supersede(old.id, new.id)
    assert memory.forget(tier=MemoryTier.PROJECT, scope="rates") == 2
    assert memory.store.get(old.id) is None


def test_a_retired_belief_stops_ageing(memory):
    """Decay ranks what is still in play; a retired belief is not."""
    old, new = memory.remember([_aged("Old rate.", 365), Belief(statement="New rate.")])
    memory.supersede(old.id, new.id)
    assert old.id not in memory.decay(half_life_days=180)


def test_the_timeline_survives_a_round_trip(memory):
    """History rides in the stored row, so it must come back with the belief."""
    belief = memory.remember([_aged("Nobody has checked this.", 180)])[0]
    memory.decay(half_life_days=180)
    causes = [r.cause for r in memory.timeline(belief.id)]
    assert causes == [Change.CREATED, Change.DECAYED]


def test_the_timeline_of_an_unknown_belief_is_empty(memory):
    assert memory.timeline("nope") == []


def test_superseding_something_that_is_not_there_changes_nothing(memory):
    real = memory.remember([Belief(statement="Real.")])[0]
    assert memory.supersede(real.id, "nope") is None
    assert memory.store.get(real.id).superseded_by is None


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
