"""The Python and Rust graphs must agree — on results and on edge cases.

Every assertion runs against both implementations. If the native build is not
present these tests skip rather than fail, because CME must stay usable as pure
Python.

Run: python tests/python_tests/test_graph_parity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.graph import KnowledgeGraph, belief_node, concept_node
from cme_python.engines.native import AVAILABLE, NativeKnowledgeGraph
from cme_python.models import Belief
from cme_python.store import BeliefStore

IMPLEMENTATIONS = [
    pytest.param(KnowledgeGraph, id="python"),
    pytest.param(
        NativeKnowledgeGraph,
        id="rust",
        marks=pytest.mark.skipif(not AVAILABLE, reason="cme_core not built"),
    ),
]


def beliefs():
    qubits = Belief(statement="Qubits are the unit of quantum information.", confidence=0.9)
    qubits.connect("Quantum Computing", "Qubits")
    superpos = Belief(statement="A qubit can hold a superposition.", confidence=0.8)
    superpos.connect("Qubits", "Superposition")
    island = Belief(statement="Rust has no garbage collector.", confidence=0.95)
    island.connect("Rust")
    return qubits, superpos, island


@pytest.fixture(params=IMPLEMENTATIONS)
def graph(request):
    qubits, superpos, island = beliefs()
    return request.param().add_all([qubits, superpos, island]), qubits, superpos, island


def test_concepts_are_the_original_labels(graph):
    g, *_ = graph
    assert g.concepts == ["Quantum Computing", "Qubits", "Rust", "Superposition"]


def test_concept_lookup_normalises(graph):
    g, qubits, superpos, _ = graph
    assert [b.id for b in g.beliefs_about("quantum  computing!")] == [qubits.id]
    assert [b.id for b in g.beliefs_about("Qubits")] == [qubits.id, superpos.id]


def test_related_finds_beliefs_sharing_a_concept(graph):
    g, qubits, superpos, island = graph
    assert [b.id for b in g.related(qubits.id)] == [superpos.id]
    assert g.related(island.id) == []
    assert g.related("ghost") == []


def test_path_traces_the_chain(graph):
    g, qubits, superpos, _ = graph
    assert g.path(belief_node(qubits), belief_node(superpos)) == [
        belief_node(qubits),
        concept_node("Qubits"),
        belief_node(superpos),
    ]


def test_unreachable_and_unknown(graph):
    g, qubits, _, island = graph
    assert g.path(belief_node(qubits), belief_node(island)) is None
    assert g.path(belief_node(qubits), belief_node("ghost")) is None


def test_max_hops_bounds_the_search(graph):
    g, qubits, superpos, _ = graph
    assert g.path(belief_node(qubits), belief_node(superpos), max_hops=1) is None
    assert g.path(belief_node(qubits), belief_node(superpos), max_hops=2) is not None


def test_path_strength(graph):
    g, qubits, superpos, _ = graph
    path = g.path(belief_node(qubits), belief_node(superpos))
    assert g.path_strength(path) == pytest.approx(0.9 * 0.8)


def test_membership_and_neighbours(graph):
    g, qubits, *_ = graph
    assert belief_node(qubits) in g
    assert concept_node("Qubits") in g
    assert concept_node("Nowhere") not in g
    assert g.neighbours(belief_node(qubits)) == {
        concept_node("Quantum Computing"),
        concept_node("Qubits"),
    }


def test_walk_reports_hop_distance(graph):
    g, qubits, superpos, _ = graph
    depths = dict(g.walk(belief_node(qubits), max_hops=2))
    assert depths[belief_node(qubits)] == 0
    assert depths[concept_node("Qubits")] == 1
    assert depths[belief_node(superpos)] == 2


def test_a_belief_with_no_connections_is_still_a_node(graph):
    g, *_ = graph
    lonely = Belief(statement="Nothing links here.")
    g.add(lonely)
    assert g.neighbours(belief_node(lonely)) == set()
    assert g.related(lonely.id) == []


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_from_store_honours_the_confidence_floor(implementation):
    with BeliefStore() as store:
        strong = Belief(statement="Kept.", confidence=0.9).connect("Topic")
        weak = Belief(statement="Dropped.", confidence=0.1).connect("Topic")
        store.save_all([strong, weak])
        g = implementation.from_store(store, min_confidence=0.5)
        assert [b.id for b in g.beliefs_about("Topic")] == [strong.id]


@pytest.mark.skipif(not AVAILABLE, reason="cme_core not built")
def test_both_implementations_agree_on_a_larger_graph():
    """Same inputs, same outputs — the guarantee that makes the port safe."""
    made = [
        Belief(statement=f"Claim {i}.", confidence=0.5 + i / 100).connect(
            f"Topic {i % 4}", f"Topic {(i + 1) % 4}"
        )
        for i in range(40)
    ]
    py = KnowledgeGraph().add_all(made)
    rs = NativeKnowledgeGraph().add_all(made)

    assert py.concepts == rs.concepts
    assert len(py) == len(rs)
    for belief in made:
        assert [b.id for b in py.related(belief.id)] == [b.id for b in rs.related(belief.id)]
    for a in made[:5]:
        for b in made[5:10]:
            p_path = py.path(belief_node(a), belief_node(b))
            r_path = rs.path(belief_node(a), belief_node(b))
            assert (p_path is None) == (r_path is None)
            if p_path is not None:
                assert len(p_path) == len(r_path)
                assert py.path_strength(p_path) == pytest.approx(rs.path_strength(r_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
