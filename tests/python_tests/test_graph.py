"""Self-check for the Knowledge Graph. Run: python tests/python_tests/test_graph.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.graph import KnowledgeGraph, belief_node, concept_node
from cme_python.models import Belief
from cme_python.store import BeliefStore


@pytest.fixture
def graph():
    """Quantum Computing -> Qubits -> Superposition, plus an unrelated island."""
    qubits = Belief(statement="Qubits are the unit of quantum information.", confidence=0.9)
    qubits.connect("Quantum Computing", "Qubits")
    superpos = Belief(statement="A qubit can hold a superposition.", confidence=0.8)
    superpos.connect("Qubits", "Superposition")
    island = Belief(statement="Rust has no garbage collector.", confidence=0.95)
    island.connect("Rust")
    g = KnowledgeGraph().add_all([qubits, superpos, island])
    return g, qubits, superpos, island


def test_concepts_keep_their_original_labels(graph):
    g, *_ = graph
    assert g.concepts == ["Quantum Computing", "Qubits", "Rust", "Superposition"]


def test_concept_lookup_ignores_case_and_punctuation(graph):
    g, qubits, superpos, _ = graph
    found = g.beliefs_about("quantum  computing!")
    assert [b.id for b in found] == [qubits.id]
    assert [b.id for b in g.beliefs_about("Qubits")] == [qubits.id, superpos.id]  # strongest first


def test_shared_concept_makes_two_beliefs_related(graph):
    g, qubits, superpos, island = graph
    assert [b.id for b in g.related(qubits.id)] == [superpos.id]
    assert g.related(island.id) == []


def test_path_traces_the_multi_hop_chain(graph):
    g, qubits, superpos, _ = graph
    path = g.path(belief_node(qubits), belief_node(superpos))
    assert path == [belief_node(qubits), concept_node("Qubits"), belief_node(superpos)]


def test_unreachable_and_unknown_nodes_return_none(graph):
    g, qubits, _, island = graph
    assert g.path(belief_node(qubits), belief_node(island)) is None
    assert g.path(belief_node(qubits), belief_node("ghost")) is None


def test_max_hops_bounds_the_search(graph):
    g, qubits, superpos, _ = graph
    assert g.path(belief_node(qubits), belief_node(superpos), max_hops=1) is None
    assert g.path(belief_node(qubits), belief_node(superpos), max_hops=2) is not None


def test_path_strength_multiplies_belief_confidence(graph):
    g, qubits, superpos, _ = graph
    path = g.path(belief_node(qubits), belief_node(superpos))
    assert g.path_strength(path) == pytest.approx(0.9 * 0.8)


def test_from_store_honours_the_confidence_floor():
    with BeliefStore() as store:
        strong = Belief(statement="Kept.", confidence=0.9).connect("Topic")
        weak = Belief(statement="Dropped.", confidence=0.1).connect("Topic")
        store.save_all([strong, weak])
        g = KnowledgeGraph.from_store(store, min_confidence=0.5)
        assert [b.id for b in g.beliefs_about("Topic")] == [strong.id]


def test_a_belief_with_no_connections_is_still_a_node():
    lonely = Belief(statement="Nothing links here.")
    g = KnowledgeGraph().add(lonely)
    assert belief_node(lonely) in g
    assert g.neighbours(belief_node(lonely)) == set()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
