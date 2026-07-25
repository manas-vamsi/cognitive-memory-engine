"""Optimization Engine — memory selection as a math problem.

Retrieval hands back the most relevant beliefs; that is not the same as the best
*set* to put in front of a model. Three beliefs saying the same thing waste the
context budget that a fourth, different one needed. Choosing the set is a
quadratic binary optimization:

    minimise   -Σ relevance_i x_i  +  λ Σ_{i<j} similarity_ij · value_ij · x_i x_j
    subject to Σ cost_i x_i ≤ budget,   x ∈ {0,1}

That is a QUBO — the exact form the Quantum Optimization Layer consumes. The
solvers here are classical; a QAOA or annealing backend swaps in behind
`Solver` and reads the same matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations, product

from cme_python.engines.evidence import EvidenceEngine, tokenise
from cme_python.models import Belief

EXHAUSTIVE_LIMIT = 18
"""Above this many candidates, 2^n brute force stops being free."""


class QUBO:
    """Quadratic Unconstrained Binary Optimization problem: minimise xᵀQx."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.terms: dict[tuple[int, int], float] = {}

    def add(self, i: int, j: int, weight: float) -> None:
        """Add to the (i, j) coefficient. i == j is the linear term."""
        key = (i, j) if i <= j else (j, i)
        self.terms[key] = self.terms.get(key, 0.0) + weight

    def energy(self, x: Sequence[int]) -> float:
        return round(
            sum(w for (i, j), w in self.terms.items() if x[i] and x[j]),
            9,
        )

    def matrix(self) -> list[list[float]]:
        """Dense upper-triangular form, for handing to an external solver."""
        m = [[0.0] * self.size for _ in range(self.size)]
        for (i, j), w in self.terms.items():
            m[i][j] = w
        return m


Feasible = Callable[[Sequence[int]], bool]
Solver = Callable[[QUBO, Feasible], list[int]]


def budget_constraint(costs: Sequence[float], budget: float) -> Feasible:
    def feasible(x: Sequence[int]) -> bool:
        return sum(c for c, on in zip(costs, x, strict=True) if on) <= budget

    return feasible


def stem(word: str) -> str:
    """Strip a plural/third-person `s` so `qubit` and `qubits` are one term."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def jaccard(a: str, b: str) -> float:
    """Overlap between two statements, 0 to 1.

    ponytail: stemmed token overlap. Without the stemmer, "Qubits can hold a
    superposition" and "A qubit holds a superposition" score 0.33 and both
    survive as separate memories — paraphrase is exactly the redundancy this
    engine exists to remove. Embedding cosine is the real fix; swap it in here.
    """
    ta = {stem(t) for t in tokenise(a)}
    tb = {stem(t) for t in tokenise(b)}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_selection_qubo(
    relevances: Sequence[float],
    similarities: dict[tuple[int, int], float],
    *,
    redundancy: float = 1.0,
) -> QUBO:
    """Encode "pick a relevant but non-repetitive set" as a QUBO.

    Relevances are normalised to [0, 1] first so `redundancy` is a unit-free
    knob: at 1.0, two identical beliefs cancel out the value of keeping both.
    """
    q = QUBO(len(relevances))
    peak = max(relevances, default=0.0) or 1.0
    scaled = [r / peak for r in relevances]
    for i, r in enumerate(scaled):
        q.add(i, i, -r)  # reward for selecting
    for (i, j), sim in similarities.items():
        if sim <= 0:
            continue
        overlap_value = (scaled[i] + scaled[j]) / 2
        q.add(i, j, redundancy * sim * overlap_value)  # penalty for repeating
    return q


# --- solvers ---------------------------------------------------------------


def solve_exhaustive(qubo: QUBO, feasible: Feasible) -> list[int]:
    """Exact ground state by enumeration. Only for small problems."""
    best: list[int] | None = None
    best_energy = 0.0  # the empty selection is always available, at energy 0
    for bits in product((0, 1), repeat=qubo.size):
        if not feasible(bits):
            continue
        e = qubo.energy(bits)
        if best is None or e < best_energy:
            best, best_energy = list(bits), e
    return best if best is not None else [0] * qubo.size


def solve_greedy(qubo: QUBO, feasible: Feasible) -> list[int]:
    """Add whichever variable lowers energy most, then try single swaps.

    ponytail: greedy plus 1-opt, no simulated annealing. It is exact on the
    small instances we can check and good enough on the rest; the quantum
    backend is the real answer for large ones.
    """
    x = [0] * qubo.size
    current = 0.0
    improved = True
    while improved:
        improved = False
        for i in range(qubo.size):
            if x[i]:
                continue
            x[i] = 1
            if feasible(x):
                e = qubo.energy(x)
                if e < current - 1e-12:
                    current = e
                    improved = True
                    continue
            x[i] = 0
    # 1-opt: swap a chosen item for an unchosen one if that helps.
    for out_i, in_j in product(range(qubo.size), repeat=2):
        if not x[out_i] or x[in_j]:
            continue
        x[out_i], x[in_j] = 0, 1
        e = qubo.energy(x)
        if feasible(x) and e < current - 1e-12:
            current = e
        else:
            x[out_i], x[in_j] = 1, 0
    return x


def solve(qubo: QUBO, feasible: Feasible) -> list[int]:
    """Exact when the problem is small enough, greedy when it is not."""
    if qubo.size <= EXHAUSTIVE_LIMIT:
        return solve_exhaustive(qubo, feasible)
    return solve_greedy(qubo, feasible)


# --- engine ----------------------------------------------------------------


class OptimizationEngine:
    """Chooses which memories to spend the context budget on."""

    def __init__(self, evidence: EvidenceEngine, solver: Solver = solve) -> None:
        self.evidence = evidence
        self.solver = solver

    def select(
        self,
        query: str,
        *,
        budget: float = 60,
        pool: int = 12,
        redundancy: float = 1.0,
        within: Callable[[Belief], bool] | None = None,
    ) -> list[Belief]:
        """The best *set* of beliefs for a query within a token budget.

        `budget` is in tokens, approximated by content-word count. `within`
        restricts the candidates to one slice of memory.
        """
        candidates = self.evidence.retrieve(query, limit=pool, within=within)
        if not candidates:
            return []
        beliefs = [b for b, _ in candidates]
        relevances = [r for _, r in candidates]
        costs = [self.cost(b) for b in beliefs]
        similarities = {
            (i, j): jaccard(beliefs[i].statement, beliefs[j].statement)
            for i, j in combinations(range(len(beliefs)), 2)
        }
        qubo = build_selection_qubo(relevances, similarities, redundancy=redundancy)
        chosen = self.solver(qubo, budget_constraint(costs, budget))
        return [b for b, on in zip(beliefs, chosen, strict=True) if on]

    @staticmethod
    def cost(belief: Belief) -> int:
        """Context cost of including a belief.

        ponytail: content-word count as a token proxy — no tokeniser
        dependency. Swap in `tiktoken` if the budget needs to be exact.
        """
        return max(len(tokenise(belief.statement)), 1)
