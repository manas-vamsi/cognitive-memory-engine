"""Self-check for the Reasoning Engine. Run: python tests/python_tests/test_reasoning.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.reasoning import ReasoningEngine, contradicts
from cme_python.models import Belief, Evidence
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
    engine.graph  # prime the cache
    fresh = Belief(statement="Entanglement correlates two qubits.", confidence=0.85)
    fresh.connect("Qubits")
    engine.store.save(fresh)
    assert engine.connect(qubits.id, fresh.id) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
