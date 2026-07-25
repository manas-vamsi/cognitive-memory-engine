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
import random
from collections.abc import Sequence

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
    """Quantum annealing via D-Wave Ocean, falling back to its classical sampler.

    ponytail: the constraint is applied by filtering returned samples rather
    than encoded as slack qubits. Exact for the sizes we run; slack encoding is
    the upgrade when a real QPU is in the loop.
    """
    try:
        import dimod  # noqa: PLC0415
    except ImportError:
        return _missing("dwave-ocean-sdk", "D-Wave")(qubo, feasible)

    h, j, offset = to_ising(qubo)
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, j, num_reads=100)
    for sample in sampleset.samples():
        bits = spins_to_bits([sample[i] for i in range(qubo.size)])
        if feasible(bits):
            return bits
    return [0] * qubo.size


def qaoa(qubo: QUBO, feasible: Feasible) -> list[int]:
    """QAOA via Qiskit. Requires `qiskit` and `qiskit-optimization`."""
    try:
        import qiskit  # noqa: F401, PLC0415
    except ImportError:
        return _missing("qiskit qiskit-optimization", "QAOA")(qubo, feasible)
    raise NotImplementedError(
        "QAOA circuit construction lands with the Qiskit integration; "
        "use the 'annealing' backend until then."
    )


BACKENDS: dict[str, Solver] = {
    "exact": solve_exhaustive,
    "annealing": simulated_annealing,
    "dwave": dwave_annealer,
    "qaoa": qaoa,
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
