"""Quantum Optimization Layer — optional acceleration for the QUBO.

Quantum does not replace the LLM and it does not replace the engines. It only
solves the optimization step the Optimization Engine already produced, so this
module is a set of interchangeable `Solver` backends over the same matrix.

Nothing here is a hard dependency. `qiskit` and `dwave-ocean-sdk` are heavy and
neither is needed to run CME, so both are imported lazily and asked for by
name. The default `annealing` backend is a classical simulated annealer using
the stdlib — the same Ising formulation a real annealer consumes, which is what
makes the comparison in research question 6 an honest one.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from typing import NamedTuple

from cme_python.engines.optimization import QUBO, Feasible, Solver, solve_exhaustive

Ising = tuple[dict[int, float], dict[tuple[int, int], float], float]
"""(h, J, offset) — fields, couplings, and the constant that restores energy."""


def to_ising(qubo: QUBO) -> Ising:
    """Convert xᵀQx over {0,1} to an Ising model over spins {-1,+1}.

    Substituting x = (1 + s) / 2 is exactly what an annealer or a QAOA circuit
    needs; `ising_energy` and `QUBO.energy` agree on every input, which is the
    property worth testing.
    """
    h: dict[int, float] = {}
    j: dict[tuple[int, int], float] = {}
    offset = 0.0
    for (a, b), w in qubo.terms.items():
        if a == b:
            h[a] = h.get(a, 0.0) + w / 2
            offset += w / 2
        else:
            j[(a, b)] = j.get((a, b), 0.0) + w / 4
            h[a] = h.get(a, 0.0) + w / 4
            h[b] = h.get(b, 0.0) + w / 4
            offset += w / 4
    return h, j, offset


def ising_energy(ising: Ising, spins: Sequence[int]) -> float:
    h, j, offset = ising
    energy = offset
    energy += sum(weight * spins[i] for i, weight in h.items())
    energy += sum(weight * spins[a] * spins[b] for (a, b), weight in j.items())
    return round(energy, 9)


def spins_to_bits(spins: Sequence[int]) -> list[int]:
    return [(s + 1) // 2 for s in spins]


def bits_to_spins(bits: Sequence[int]) -> list[int]:
    return [2 * b - 1 for b in bits]


# --- classical stand-in ----------------------------------------------------


def simulated_annealing(
    qubo: QUBO,
    feasible: Feasible,
    *,
    sweeps: int = 400,
    restarts: int = 8,
    seed: int = 0,
) -> list[int]:
    """Classical annealer over the Ising form — the quantum backend's stand-in.

    Deterministic by default so tests and benchmarks are reproducible; pass a
    different `seed` to sample.

    ponytail: geometric cooling, single-spin flips. Enough to match the exact
    solver on the instances we can verify. Real hardware is the upgrade path,
    which is the whole point of this module.
    """
    if qubo.size == 0:
        return []
    rng = random.Random(seed)
    best = [0] * qubo.size
    best_energy = 0.0 if feasible(best) else math.inf

    for restart in range(restarts):
        current = [0] * qubo.size
        if not feasible(current):
            continue
        energy = qubo.energy(current)
        for step in range(sweeps):
            # Cool from accepting most uphill moves to accepting almost none.
            temperature = max(1e-6, 1.0 * (1 - step / sweeps) ** 2)
            i = rng.randrange(qubo.size)
            current[i] ^= 1
            if not feasible(current):
                current[i] ^= 1
                continue
            candidate = qubo.energy(current)
            delta = candidate - energy
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                energy = candidate
            else:
                current[i] ^= 1
        if energy < best_energy:
            best, best_energy = list(current), energy
        rng.seed(seed + restart + 1)
    return best


# --- optional hardware backends --------------------------------------------


def _missing(package: str, extra: str) -> Solver:
    def solver(qubo: QUBO, feasible: Feasible) -> list[int]:
        raise RuntimeError(
            f"The {extra} backend needs `{package}`, which CME does not install by "
            f"default. Run `pip install {package}`, or use the 'annealing' backend."
        )

    return solver


def dwave_annealer(qubo: QUBO, feasible: Feasible) -> list[int]:
    """Quantum annealing via D-Wave Ocean.

    On a real QPU when `DWAVE_API_TOKEN` is set and `dwave-system` is
    installed, and on Ocean's classical sampler otherwise, so the same code path
    runs in CI and on hardware.

    ponytail: the constraint is applied by filtering returned samples rather
    than encoded as slack qubits. Exact for the sizes we run; slack encoding is
    the upgrade when the QPU returns mostly over-budget samples.
    """
    try:
        import dimod  # noqa: PLC0415
    except ImportError:
        return _missing("dwave-ocean-sdk", "D-Wave")(qubo, feasible)

    h, j, _ = to_ising(qubo)
    if os.environ.get("DWAVE_API_TOKEN"):
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite  # noqa: PLC0415
        except ImportError:
            return _missing("dwave-system", "D-Wave QPU")(qubo, feasible)
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, j, num_reads=100)
    for sample in sampleset.samples():  # lowest energy first
        bits = spins_to_bits([sample[i] for i in range(qubo.size)])
        if feasible(bits):
            return bits
    return [0] * qubo.size


STATEVECTOR_LIMIT = 16
"""Variables above which simulating the backends below stops being reasonable.

Both keep a table of all 2^n states: 65,536 at 16, a million at 20. That is the
simulator's ceiling, not the algorithm's; on hardware the table does not exist.
"""


def energy_table(qubo: QUBO, feasible: Feasible):
    """Energy of every one of the 2^n selections, infeasible ones at +inf.

    Index i is the selection whose bit q is `(i >> q) & 1`, which is Qiskit's
    qubit order, so the table lines up with a statevector without reshuffling.
    """
    import numpy as np  # noqa: PLC0415

    n = qubo.size
    if n > STATEVECTOR_LIMIT:
        raise ValueError(
            f"{n} variables is past the {STATEVECTOR_LIMIT} a statevector simulation "
            "can hold. Use a smaller pool, the 'annealing' backend, or hardware."
        )
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    energies = np.einsum("ki,ij,kj->k", bits, np.array(qubo.matrix()), bits).round(9)
    allowed = np.fromiter((feasible(row) for row in bits.tolist()), bool, 2**n)
    return np.where(allowed, energies, np.inf)


def _bits(index: int, size: int) -> list[int]:
    return [(index >> q) & 1 for q in range(size)]


def cost_hamiltonian(qubo: QUBO):
    """The QUBO as a Pauli-Z Hamiltonian, for a circuit to minimise.

    From the Ising form: x = (1 + s) / 2 there, and Z reads +1 on |0>, so each
    spin is -Z. Fields flip sign, couplings do not. The constant offset is
    dropped because it moves every energy equally.
    """
    from qiskit.quantum_info import SparsePauliOp  # noqa: PLC0415

    h, j, _ = to_ising(qubo)
    terms = [("Z", [i], -w) for i, w in h.items() if w]
    terms += [("ZZ", [a, b], w) for (a, b), w in j.items() if w]
    return SparsePauliOp.from_sparse_list(terms or [("I", [0], 0.0)], num_qubits=qubo.size)


def qaoa(
    qubo: QUBO,
    feasible: Feasible,
    *,
    reps: int = 2,
    restarts: int = 3,
    maxiter: int = 100,
    shots: int = 1024,
    seed: int = 0,
) -> list[int]:
    """QAOA via Qiskit, simulated on a statevector.

    A real Qiskit circuit, `qaoa_ansatz` over the cost Hamiltonian, so running
    it on hardware means swapping the simulator for a sampler and nothing else.
    COBYLA tunes the angles, `maxiter` evaluations per restart: its default of
    a thousand took 50 seconds at 12 qubits for no better answer. The objective
    is the expected energy with over-budget selections priced above any
    affordable one, which is how the constraint reaches a circuit that cannot
    encode it. Then `shots` are drawn
    from the final state and the best affordable one measured is the answer,
    which is exactly what a device would report.

    ponytail: constraint by penalty in the classical loop, not in the circuit.
    Encoding the budget as slack qubits is the upgrade, and needs more qubits
    than the pool has to spare.
    """
    if qubo.size == 0:
        return []
    try:
        import numpy as np  # noqa: PLC0415
        from qiskit.circuit.library import qaoa_ansatz  # noqa: PLC0415
        from qiskit.quantum_info import Statevector  # noqa: PLC0415
        from scipy.optimize import minimize  # noqa: PLC0415
    except ImportError:
        return _missing("qiskit scipy", "QAOA")(qubo, feasible)

    energies = energy_table(qubo, feasible)
    affordable = energies[np.isfinite(energies)]
    ceiling = affordable.max() + (affordable.max() - affordable.min()) + 1.0
    priced = np.where(np.isfinite(energies), energies, ceiling)
    circuit = qaoa_ansatz(cost_hamiltonian(qubo), reps=reps)

    def probabilities(angles):
        return Statevector(circuit.assign_parameters(angles)).probabilities()

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        start = rng.uniform(0, np.pi, circuit.num_parameters)
        run = minimize(
            lambda a: float(probabilities(a) @ priced),
            start,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        if best is None or run.fun < best.fun:
            best = run
    final = probabilities(best.x)
    measured = set(rng.choice(final.size, size=shots, p=final / final.sum()).tolist())
    answer = min(measured, key=lambda i: (energies[i], i))
    return _bits(answer, qubo.size) if np.isfinite(energies[answer]) else [0] * qubo.size


class GroverRun(NamedTuple):
    index: int
    """The selection it settled on."""
    found_at: int
    """Oracle calls spent when that selection was first measured."""
    spent: int
    """Oracle calls spent in all, which is what the guarantee costs."""


def durr_hoyer(energies, *, seed: int = 0) -> GroverRun:
    """Grover minimum finding (Durr and Hoyer, 1996).

    Keep a threshold, the best energy seen. The oracle marks every state below
    it; Grover amplification makes a marked state likely to be measured; measure,
    and a success lowers the threshold. The number of amplification rounds is
    unknown in advance, so each attempt draws it at random from a range that
    grows by 6/5 on failure (Boyer, Brassard, Hoyer and Tapp).

    Simulated exactly rather than by building a circuit: with M of N states
    marked, k rounds succeed with probability sin^2((2k + 1) theta), where
    sin(theta) = sqrt(M / N), and the state measured on success is uniform over
    the marked ones. That is the algorithm's full measurement statistics, which
    is all a circuit would add, at none of its cost. Oracle calls are counted the
    way the algorithm spends them, and stop at the 22.5 sqrt(N) bound under
    which it finds the minimum with probability at least 1/2.

    Two counts, because they answer different questions. `found_at` is what
    reaching the minimum took. `spent` is what the algorithm pays, since it
    cannot know it is done: 22.5 sqrt(N) is 1,440 calls at 12 qubits, a third
    of the 4,096 states. The advantage is real but asymptotic, and at the sizes
    a context pool has, that constant is most of the story.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    size = energies.size
    affordable = np.flatnonzero(np.isfinite(energies))
    if affordable.size == 0:
        return GroverRun(0, 0, 0)
    best = int(rng.choice(affordable))
    found_at = 0
    calls, limit, growth = 0, 22.5 * math.sqrt(size), 1.0
    while calls < limit:
        marked = np.flatnonzero(energies < energies[best] - 1e-12)
        rounds = int(rng.integers(0, int(growth) + 1))
        calls += rounds + 1  # the rounds, plus the check of what was measured
        theta = math.asin(math.sqrt(marked.size / size))
        if marked.size and rng.random() < math.sin((2 * rounds + 1) * theta) ** 2:
            best, growth, found_at = int(rng.choice(marked)), 1.0, calls
        else:
            growth = min(growth * 6 / 5, math.sqrt(size))
    return GroverRun(best, found_at, calls)


def grover_search(qubo: QUBO, feasible: Feasible, *, seed: int = 0) -> list[int]:
    """Grover-inspired search: the lowest-energy affordable selection, by `durr_hoyer`.

    Quadratically fewer oracle calls than checking every selection, which is
    the whole claim, and `benchmarks/run.py` counts them rather than asserting it.
    """
    if qubo.size == 0:
        return []
    try:
        import numpy  # noqa: F401, PLC0415
    except ImportError:
        return _missing("numpy", "Grover")(qubo, feasible)
    energies = energy_table(qubo, feasible)
    index = durr_hoyer(energies, seed=seed).index
    return _bits(index, qubo.size) if math.isfinite(energies[index]) else [0] * qubo.size


BACKENDS: dict[str, Solver] = {
    "exact": solve_exhaustive,
    "annealing": simulated_annealing,
    "dwave": dwave_annealer,
    "qaoa": qaoa,
    "grover": grover_search,
}


def get_solver(name: str = "annealing") -> Solver:
    """Look up a solver backend by name.

    Hand the result straight to `OptimizationEngine(evidence, solver=...)` —
    every backend reads the same QUBO, which is what lets them be compared.
    """
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {', '.join(sorted(BACKENDS))}"
        ) from None
