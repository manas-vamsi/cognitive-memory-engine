"""Self-check for the Reasoning Engine. Run: python tests/python_tests/test_reasoning.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.reasoning import ReasoningEngine, contradicts
from cme_python.models import Belief, Change, Evidence
from cme_python.store import BeliefStore

HAS_GC = "Rust has a garbage collector."
NO_GC = "Rust has no garbage collector."


def test_the_same_claim_asserted_and_denied_is_a_contradiction():
    assert contradicts(HAS_GC, NO_GC) > 0.9


def test_agreeing_and_unrelated_statements_are_not_contradictions():
    assert contradicts(HAS_GC, HAS_GC) == 0.0
    assert contradicts(NO_GC, "Rust does not have a garbage collector.") == 0.0
    assert contradicts(NO_GC, "Qubits can hold a superposition.") == 0.0


def test_negation_about_a_different_subject_is_not_a_contradiction():
    assert contradicts("Python has a garbage collector.", NO_GC) == 0.0


def test_double_negation_reads_as_affirmative():
    assert contradicts("It is not true that Rust has no garbage collector.", NO_GC) > 0.0


def test_empty_statements_are_handled():
    assert contradicts("", NO_GC) == 0.0


# --- engine ----------------------------------------------------------------


@pytest.fixture
def setup():
    """Two beliefs sharing the Qubits concept, plus a contradicting pair."""
    with BeliefStore() as store:
        qubits = Belief(statement="Qubits are the unit of quantum information.", confidence=0.9)
        qubits.connect("Quantum Computing", "Qubits")
        superpos = Belief(statement="A qubit can hold a superposition.", confidence=0.8)
        superpos.connect("Qubits")
        yes = Belief(statement=HAS_GC, confidence=0.6)
        no = Belief(statement=NO_GC, confidence=0.7)
        no.add_evidence(Evidence(snippet="The Rust Book, ownership chapter", strength=0.9))
        store.save_all([qubits, superpos, yes, no])
        yield ReasoningEngine(store), qubits, superpos, yes, no


@pytest.fixture
def store_with_close_clash():
    """A contradiction where neither side is better evidenced than the other."""
    with BeliefStore() as store:
        weaker = Belief(statement=HAS_GC, confidence=0.60)
        stronger = Belief(statement=NO_GC, confidence=0.64)
        store.save_all([weaker, stronger])
        yield ReasoningEngine(store), weaker, stronger


def test_connect_returns_the_chain_through_the_shared_concept(setup):
    engine, qubits, superpos, *_ = setup
    chain = engine.connect(qubits.id, superpos.id)
    assert chain is not None
    assert [b.id for b in chain.beliefs] == [qubits.id, superpos.id]
    assert chain.concepts == ["qubits"]
    assert chain.hops == 1
    assert chain.strength == pytest.approx(0.9 * 0.8)
    assert "superposition" in chain.explain()


def test_connect_returns_none_for_unrelated_beliefs(setup):
    engine, qubits, _, yes, _ = setup
    assert engine.connect(qubits.id, yes.id) is None
    assert engine.connect(qubits.id, "ghost") is None


def test_infer_ranks_chains_by_strength(setup):
    engine, qubits, superpos, *_ = setup
    chains = engine.infer(qubits.id)
    assert [c.beliefs[-1].id for c in chains] == [superpos.id]


def test_contradictions_finds_the_clashing_pair_only(setup):
    engine, _, _, yes, no = setup
    found = engine.contradictions()
    assert len(found) == 1
    assert {found[0].a.id, found[0].b.id} == {yes.id, no.id}
    assert "contradicts" in found[0].explain()


def test_the_better_evidenced_side_wins_and_the_other_is_weakened(setup):
    engine, _, _, yes, no = setup
    clash = engine.contradictions()[0]
    assert clash.winner.id == no.id  # it carries supporting evidence
    assert clash.loser.id == yes.id

    weakened = engine.resolve(clash)
    assert weakened.confidence < 0.6
    assert engine.store.get(yes.id).confidence == weakened.confidence  # persisted
    assert engine.store.get(no.id).confidence == no.confidence  # winner untouched


def test_nothing_is_deleted_when_a_contradiction_is_resolved(setup):
    engine, *_ = setup
    engine.resolve(engine.contradictions()[0])
    assert len(engine.store) == 4


def test_resolving_writes_the_reason_into_the_timeline(setup):
    """A confidence that moved with no record of why is a number nobody can audit."""
    engine, _, _, yes, no = setup
    weakened = engine.resolve(engine.contradictions()[0])
    assert weakened.history[-1].cause == Change.CONTRADICTED
    assert weakened.history[-1].note == no.id


# --- reconciliation ----------------------------------------------------------


def test_reconcile_retires_the_decisively_beaten_side(setup):
    """One side has a source and the other has nothing. That is not a close call."""
    engine, _, _, yes, no = setup
    done = engine.reconcile()
    assert len(done) == 1
    assert done[0].retired is True
    assert done[0].winner == no.id and done[0].loser == yes.id

    assert engine.store.get(yes.id).superseded_by == no.id
    assert [b.id for b in engine.store.all()] != []  # the winner is still there
    assert yes.id not in [b.id for b in engine.store.all()]


def test_reconcile_only_weakens_a_close_call(store_with_close_clash):
    """A narrow margin is a reason to trust a claim less, not to retire it."""
    engine, weaker, stronger = store_with_close_clash
    done = engine.reconcile()
    assert len(done) == 1
    assert done[0].retired is False
    assert done[0].margin < 0.35

    survivor = engine.store.get(weaker.id)
    assert survivor.superseded_by is None  # still in play
    assert survivor.confidence < weaker.confidence  # but trusted less
    assert survivor.history[-1].cause == Change.CONTRADICTED


def test_reconcile_will_not_break_a_dead_heat():
    """Equal backing gives no reason to prefer either side; a coin toss is not inference."""
    with BeliefStore() as store:
        a = Belief(statement=HAS_GC, confidence=0.7)
        b = Belief(statement=NO_GC, confidence=0.7)
        store.save_all([a, b])
        engine = ReasoningEngine(store)

        assert engine.reconcile() == []
        assert len(engine.contradictions()) == 1  # still flagged for a human
        assert store.get(a.id).confidence == 0.7
        assert store.get(b.id).confidence == 0.7


def test_reconcile_leaves_a_consistent_registry_alone(setup):
    engine, *_ = setup
    engine.reconcile()
    assert engine.reconcile() == []


def test_reconcile_never_deletes(setup):
    """Retired is not deleted: the timeline is why the survivor is trusted."""
    engine, *_ = setup
    engine.reconcile()
    assert len(engine.store) == 4


def test_propagation_records_why_a_neighbour_moved(setup):
    engine, qubits, superpos, *_ = setup
    engine.propagate(qubits.id, -0.2)
    moved = engine.store.get(superpos.id)
    assert moved.history[-1].cause == Change.PROPAGATED
    assert qubits.id in moved.history[-1].note


def test_propagation_moves_neighbours_less_than_the_source(setup):
    engine, qubits, superpos, *_ = setup
    before = superpos.confidence
    moved = engine.propagate(qubits.id, -0.3)

    assert moved[qubits.id] == pytest.approx(0.6)
    assert superpos.id in moved
    assert moved[superpos.id] < before  # dropped
    assert before - moved[superpos.id] < 0.3  # but by less than the source


def test_propagation_leaves_unconnected_beliefs_alone(setup):
    engine, qubits, _, yes, _ = setup
    moved = engine.propagate(qubits.id, -0.3)
    assert yes.id not in moved
    assert engine.store.get(yes.id).confidence == 0.6


def test_propagation_persists_and_stays_in_range(setup):
    engine, qubits, superpos, *_ = setup
    engine.propagate(qubits.id, 5.0)  # absurd shove
    assert engine.store.get(qubits.id).confidence == 1.0
    assert 0.0 <= engine.store.get(superpos.id).confidence <= 1.0


def test_propagating_an_unknown_belief_is_a_no_op(setup):
    engine, *_ = setup
    assert engine.propagate("ghost", -0.5) == {}


def test_the_graph_refreshes_after_the_registry_changes(setup):
    engine, qubits, *_ = setup
    assert engine.graph is not None  # prime the cache
    fresh = Belief(statement="Entanglement correlates two qubits.", confidence=0.85)
    fresh.connect("Qubits")
    engine.store.save(fresh)
    assert engine.connect(qubits.id, fresh.id) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
