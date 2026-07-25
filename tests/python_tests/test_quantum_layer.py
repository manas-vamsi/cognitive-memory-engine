"""Self-check for the Quantum Optimization Layer.

Run: python tests/python_tests/test_quantum_layer.py
"""

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.evidence import EvidenceEngine
from cme_python.engines.optimization import (
    QUBO,
    OptimizationEngine,
    budget_constraint,
    build_selection_qubo,
    solve_exhaustive,
)
from cme_python.engines.quantum_layer import (
    BACKENDS,
    bits_to_spins,
    get_solver,
    ising_energy,
    simulated_annealing,
    spins_to_bits,
    to_ising,
)
from cme_python.models import Belief
from cme_python.store import BeliefStore

ALWAYS = lambda x: True  # noqa: E731


def a_problem() -> QUBO:
    sims = {(0, 1): 0.9, (0, 2): 0.1, (1, 2): 0.4}
    return build_selection_qubo([1.0, 0.8, 0.6], sims, redundancy=1.3)


def test_spin_and_bit_conversion_round_trips():
    for bits in product((0, 1), repeat=3):
        assert spins_to_bits(bits_to_spins(bits)) == list(bits)


def test_ising_energy_matches_qubo_energy_everywhere():
    """The conversion is only useful if it preserves the objective exactly."""
    qubo = a_problem()
    ising = to_ising(qubo)
    for bits in product((0, 1), repeat=qubo.size):
        assert ising_energy(ising, bits_to_spins(bits)) == pytest.approx(qubo.energy(bits))


def test_ising_of_an_empty_problem_is_empty():
    h, j, offset = to_ising(QUBO(0))
    assert (h, j, offset) == ({}, {}, 0.0)


def test_annealing_finds_the_exact_ground_state():
    qubo = a_problem()
    assert qubo.energy(simulated_annealing(qubo, ALWAYS)) == pytest.approx(
        qubo.energy(solve_exhaustive(qubo, ALWAYS))
    )


def test_annealing_respects_the_budget():
    qubo = a_problem()
    feasible = budget_constraint([4, 4, 4], budget=5)
    chosen = simulated_annealing(qubo, feasible)
    assert feasible(chosen)
    assert sum(chosen) == 1


def test_annealing_is_deterministic_for_a_given_seed():
    qubo = a_problem()
    assert simulated_annealing(qubo, ALWAYS, seed=7) == simulated_annealing(qubo, ALWAYS, seed=7)


def test_annealing_handles_an_empty_problem():
    assert simulated_annealing(QUBO(0), ALWAYS) == []


def test_annealing_returns_nothing_when_every_move_is_infeasible():
    qubo = a_problem()
    assert simulated_annealing(qubo, budget_constraint([9, 9, 9], budget=1)) == [0, 0, 0]


def test_get_solver_by_name_and_unknown_name_is_reported():
    assert get_solver("annealing") is simulated_annealing
    assert get_solver("exact") is solve_exhaustive
    with pytest.raises(ValueError, match="Unknown backend"):
        get_solver("teleportation")


def test_missing_hardware_backends_say_how_to_install_them():
    """Absent qiskit/ocean must be a clear message, not an ImportError."""
    for name in ("dwave", "qaoa"):
        try:
            BACKENDS[name](a_problem(), ALWAYS)
        except NotImplementedError as exc:  # library present, circuit not built yet
            assert "annealing" in str(exc)  # NB: subclasses RuntimeError, so catch it first
        except RuntimeError as exc:
            assert "pip install" in str(exc)


def test_the_engine_accepts_a_quantum_backend_and_agrees_with_the_default():
    with BeliefStore() as store:
        store.save_all(
            [
                Belief(statement="Qubits can hold a superposition of states.", confidence=0.9),
                Belief(statement="A qubit holds a superposition of states.", confidence=0.9),
                Belief(statement="Entanglement correlates two separated qubits.", confidence=0.85),
            ]
        )
        evidence = EvidenceEngine(store)
        query = "qubits superposition entanglement"
        classical = OptimizationEngine(evidence).select(query)
        annealed = OptimizationEngine(evidence, solver=get_solver("annealing")).select(query)
        assert {b.id for b in annealed} == {b.id for b in classical}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
