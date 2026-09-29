"""Self-check for the Quantum Optimization Layer.

Run: python tests/python_tests/test_quantum_layer.py
"""

import random
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
    STATEVECTOR_LIMIT,
    bits_to_spins,
    cost_hamiltonian,
    durr_hoyer,
    energy_table,
    get_solver,
    grover_search,
    ising_energy,
    qaoa,
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


def test_every_optional_backend_solves_or_says_how_to_install():
    """Absent qiskit/ocean/numpy must be a clear message, not an ImportError."""
    qubo = a_problem()
    for name in ("dwave", "qaoa", "grover"):
        try:
            chosen = BACKENDS[name](qubo, ALWAYS)
        except RuntimeError as exc:
            assert "pip install" in str(exc)
        else:
            assert qubo.energy(chosen) == pytest.approx(qubo.energy(solve_exhaustive(qubo, ALWAYS)))


def random_problem(seed: int, n: int) -> tuple[QUBO, object]:
    rng = random.Random(seed)
    sims = {(i, j): rng.random() * 0.6 for i in range(n) for j in range(i + 1, n)}
    qubo = build_selection_qubo([rng.random() for _ in range(n)], sims, redundancy=1.2)
    return qubo, budget_constraint([rng.randint(2, 6) for _ in range(n)], budget=12)


def test_the_cost_hamiltonian_is_the_qubo_on_every_state():
    """A circuit minimising the wrong operator would optimise the wrong thing."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("qiskit")
    qubo, _ = random_problem(0, 5)
    diagonal = np.real(np.diag(cost_hamiltonian(qubo).to_matrix())) + to_ising(qubo)[2]
    assert np.allclose(diagonal, energy_table(qubo, ALWAYS))


def test_the_energy_table_follows_qiskit_qubit_order():
    pytest.importorskip("numpy")
    qubo = QUBO(3)
    qubo.add(0, 0, -1.0)  # only bit 0 matters
    table = energy_table(qubo, ALWAYS)
    assert [i for i in range(8) if table[i] == -1.0] == [1, 3, 5, 7]  # bit 0 = index & 1


def test_infeasible_states_are_priced_out_of_the_table():
    pytest.importorskip("numpy")
    table = energy_table(a_problem(), budget_constraint([4, 4, 4], budget=5))
    assert all(table[i] == float("inf") for i in (3, 5, 6, 7))  # two or more items


def test_the_simulators_refuse_a_problem_too_big_to_hold():
    pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="statevector"):
        energy_table(QUBO(STATEVECTOR_LIMIT + 1), ALWAYS)


def test_qaoa_finds_the_ground_state_within_the_budget():
    pytest.importorskip("qiskit")
    pytest.importorskip("scipy")
    qubo, feasible = random_problem(1, 6)
    chosen = qaoa(qubo, feasible)
    assert feasible(chosen)
    assert qubo.energy(chosen) == pytest.approx(qubo.energy(solve_exhaustive(qubo, feasible)))


def test_grover_finds_the_ground_state_within_the_budget():
    pytest.importorskip("numpy")
    for seed in range(5):
        qubo, feasible = random_problem(seed, 10)
        chosen = grover_search(qubo, feasible, seed=seed)
        assert feasible(chosen)
        assert qubo.energy(chosen) == pytest.approx(qubo.energy(solve_exhaustive(qubo, feasible)))


def test_grover_reaches_the_minimum_in_far_fewer_calls_than_there_are_states():
    """The quadratic claim: well under N checks, where classical search needs N."""
    pytest.importorskip("numpy")
    qubo, feasible = random_problem(3, 12)
    table = energy_table(qubo, feasible)
    run = durr_hoyer(table, seed=3)
    assert table[run.index] == table.min()
    assert run.found_at < table.size / 10


def test_grover_pays_its_guarantee_in_full():
    """It cannot know it is done, so it spends the 22.5 sqrt(N) bound regardless."""
    pytest.importorskip("numpy")
    qubo, feasible = random_problem(3, 12)
    run = durr_hoyer(energy_table(qubo, feasible), seed=3)
    assert 22.5 * 64 <= run.spent < 22.5 * 64 + 64 + 1  # the bound, overshot by one attempt


def test_quantum_backends_handle_an_empty_problem():
    assert grover_search(QUBO(0), ALWAYS) == []
    assert qaoa(QUBO(0), ALWAYS) == []


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
