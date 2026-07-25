"""Self-check for the Optimization Engine. Run: python tests/python_tests/test_optimization.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.evidence import EvidenceEngine
from cme_python.engines.optimization import (
    QUBO,
    OptimizationEngine,
    budget_constraint,
    build_selection_qubo,
    jaccard,
    solve_exhaustive,
    solve_greedy,
)
from cme_python.models import Belief
from cme_python.store import BeliefStore

ALWAYS = lambda x: True  # noqa: E731


def test_qubo_energy_counts_linear_and_quadratic_terms():
    q = QUBO(2)
    q.add(0, 0, -1.0)
    q.add(1, 1, -1.0)
    q.add(0, 1, 3.0)
    assert q.energy([0, 0]) == 0.0
    assert q.energy([1, 0]) == -1.0
    assert q.energy([1, 1]) == 1.0  # the pair penalty outweighs both rewards


def test_add_accumulates_and_is_order_insensitive():
    q = QUBO(2)
    q.add(0, 1, 1.5)
    q.add(1, 0, 0.5)
    assert q.terms == {(0, 1): 2.0}


def test_matrix_is_upper_triangular():
    q = QUBO(2)
    q.add(1, 0, 2.0)
    assert q.matrix() == [[0.0, 2.0], [0.0, 0.0]]


def test_jaccard_bounds():
    assert jaccard("qubits hold superposition", "qubits hold superposition") == 1.0
    assert jaccard("rust memory safety", "cheese orbits jupiter") == 0.0
    assert jaccard("", "anything") == 0.0


def test_jaccard_sees_through_plurals():
    """Paraphrase is the redundancy worth catching, so stems must match."""
    assert (
        jaccard(
            "Qubits can hold a superposition of states.", "A qubit holds a superposition of states."
        )
        == 1.0
    )


def test_budget_constraint_rejects_overspend():
    feasible = budget_constraint([3, 4], budget=5)
    assert feasible([1, 0]) and feasible([0, 1])
    assert not feasible([1, 1])


def test_redundant_pair_is_split_up():
    """Two identical beliefs and one different: keep one of the pair, plus the odd one."""
    sims = {(0, 1): 1.0, (0, 2): 0.0, (1, 2): 0.0}
    qubo = build_selection_qubo([1.0, 1.0, 0.6], sims, redundancy=1.0)
    chosen = solve_exhaustive(qubo, ALWAYS)
    assert chosen[2] == 1
    assert chosen[0] + chosen[1] == 1  # not both


def test_redundancy_zero_keeps_everything_relevant():
    sims = {(0, 1): 1.0}
    qubo = build_selection_qubo([1.0, 1.0], sims, redundancy=0.0)
    assert solve_exhaustive(qubo, ALWAYS) == [1, 1]


def test_irrelevant_candidates_are_left_out():
    qubo = build_selection_qubo([1.0, 0.0], {}, redundancy=1.0)
    assert solve_exhaustive(qubo, ALWAYS) == [1, 0]


def test_budget_forces_a_choice_between_two_good_beliefs():
    qubo = build_selection_qubo([1.0, 0.8], {(0, 1): 0.0}, redundancy=1.0)
    chosen = solve_exhaustive(qubo, budget_constraint([5, 5], budget=5))
    assert chosen == [1, 0]  # the more relevant one wins the single slot


def test_greedy_matches_exhaustive_on_small_instances():
    sims = {(0, 1): 0.9, (0, 2): 0.1, (1, 2): 0.2, (0, 3): 0.0, (1, 3): 0.5, (2, 3): 0.3}
    qubo = build_selection_qubo([1.0, 0.9, 0.7, 0.4], sims, redundancy=1.2)
    feasible = budget_constraint([4, 4, 4, 4], budget=12)
    assert qubo.energy(solve_greedy(qubo, feasible)) == pytest.approx(
        qubo.energy(solve_exhaustive(qubo, feasible))
    )


def test_greedy_can_miss_the_ground_state_but_annealing_does_not():
    """Why `solve` dispatches to annealing above the exhaustive limit.

    Greedy is the faster solver and the wrong default: selecting the wrong
    memories quickly is still the wrong answer. Measured in benchmarks/run.py.
    """
    from cme_python.engines.quantum_layer import simulated_annealing

    n = 8
    relevances = [0.5 + (i % 7) / 10 for i in range(n)]
    statements = [f"claim {i % 5} variant {i}" for i in range(n)]
    sims = {
        (i, j): jaccard(statements[i], statements[j]) for i in range(n) for j in range(i + 1, n)
    }
    qubo = build_selection_qubo(relevances, sims, redundancy=1.2)
    feasible = budget_constraint([3] * n, budget=12)

    best = qubo.energy(solve_exhaustive(qubo, feasible))
    assert qubo.energy(solve_greedy(qubo, feasible)) > best + 1e-6  # greedy settles early
    assert qubo.energy(simulated_annealing(qubo, feasible)) == pytest.approx(best)


def test_solve_uses_annealing_beyond_the_exhaustive_limit():
    from cme_python.engines.optimization import EXHAUSTIVE_LIMIT, solve

    n = EXHAUSTIVE_LIMIT + 2
    qubo = build_selection_qubo([0.9] * n, {}, redundancy=1.0)
    chosen = solve(qubo, budget_constraint([1] * n, budget=n))
    assert sum(chosen) == n  # nothing penalised, so everything is worth taking


def test_empty_problem_is_handled():
    assert solve_exhaustive(QUBO(0), ALWAYS) == []
    assert solve_greedy(QUBO(0), ALWAYS) == []


def test_infeasible_problem_selects_nothing():
    qubo = build_selection_qubo([1.0], {}, redundancy=1.0)
    feasible = budget_constraint([10], budget=1)
    assert solve_exhaustive(qubo, feasible) == [0]
    assert solve_greedy(qubo, feasible) == [0]


# --- end to end ------------------------------------------------------------


@pytest.fixture
def engine():
    with BeliefStore() as store:
        store.save_all(
            [
                Belief(statement="Qubits can hold a superposition of states.", confidence=0.9),
                Belief(statement="A qubit holds a superposition of states.", confidence=0.9),
                Belief(statement="Entanglement correlates two separated qubits.", confidence=0.85),
                Belief(statement="Bread dough needs yeast to rise.", confidence=0.9),
            ]
        )
        yield OptimizationEngine(EvidenceEngine(store))


def test_selection_drops_the_near_duplicate(engine):
    picked = [b.statement for b in engine.select("qubits superposition entanglement")]
    assert "Entanglement correlates two separated qubits." in picked
    # The two superposition statements say the same thing; only one earns a slot.
    assert sum("superposition" in s for s in picked) == 1


def test_selection_respects_the_budget(engine):
    picked = engine.select("qubits superposition entanglement", budget=6)
    assert 0 < sum(engine.cost(b) for b in picked) <= 6


def test_selection_ignores_unrelated_memories(engine):
    picked = [b.statement for b in engine.select("qubits superposition")]
    assert "Bread dough needs yeast to rise." not in picked


def test_no_candidates_selects_nothing(engine):
    assert engine.select("cricket scores in 1998") == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
