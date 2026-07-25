"""Measure the claims this project makes about itself.

Three of them are currently assertions rather than numbers:

  1. The Rust core is faster than the Python graph  (README says so)
  2. Optimised selection beats a naive top-k slice  (research question 4)
  3. Quantum-style solvers rival exact search       (research question 6)

Run: python benchmarks/run.py [--scale 400]

Timings are wall-clock on one machine and will differ on yours; the shapes are
what matter. Everything here is deterministic apart from the timings themselves.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cme_python.engines.evidence import EvidenceEngine
from cme_python.engines.graph import KnowledgeGraph, belief_node
from cme_python.engines.native import AVAILABLE, NativeKnowledgeGraph
from cme_python.engines.optimization import (
    OptimizationEngine,
    budget_constraint,
    build_selection_qubo,
    jaccard,
    solve_exhaustive,
    solve_greedy,
)
from cme_python.engines.quantum_layer import simulated_annealing
from cme_python.models import Belief
from cme_python.store import BeliefStore

TOPICS = ["Quantum Computing", "Rust", "Machine Learning", "Databases", "Networking"]


def synthetic_beliefs(n: int) -> list[Belief]:
    """A connected graph with realistic fan-out: each belief touches 2 topics."""
    made = []
    for i in range(n):
        b = Belief(
            statement=f"Synthetic claim number {i} about a subject.",
            confidence=0.5 + (i % 50) / 100,
        )
        b.connect(TOPICS[i % len(TOPICS)], TOPICS[(i + 1) % len(TOPICS)])
        made.append(b)
    return made


def timed(fn, repeats: int = 5) -> float:
    """Best-of-N milliseconds. Best, not mean — it is the least noisy estimate."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return min(times)


# --- 1. graph backends ------------------------------------------------------


def bench_graph(scale: int) -> None:
    print(f"\n## Graph traversal ({scale} beliefs)\n")
    if not AVAILABLE:
        print("  cme_core not built — skipping. Build it to compare:")
        print("  cd cme-core && maturin build --release --features extension-module --out dist")
        return

    beliefs = synthetic_beliefs(scale)
    ids = [b.id for b in beliefs]

    def build(cls):
        return lambda: cls().add_all(beliefs)

    py_graph = KnowledgeGraph().add_all(beliefs)
    rs_graph = NativeKnowledgeGraph().add_all(beliefs)

    def traverse(g):
        def run():
            for start in ids[:50]:
                g.related(start, max_hops=2)
            for a, b in zip(ids[:25], ids[25:50], strict=True):
                g.path(belief_node(a), belief_node(b))

        return run

    rows = [
        ("build", timed(build(KnowledgeGraph)), timed(build(NativeKnowledgeGraph))),
        ("traverse", timed(traverse(py_graph)), timed(traverse(rs_graph))),
    ]
    print(f"  {'operation':<12} {'python':>10} {'rust':>10} {'speedup':>10}")
    for name, py, rs in rows:
        ratio = py / rs if rs else float("inf")
        print(f"  {name:<12} {py:>9.2f}ms {rs:>9.2f}ms {ratio:>9.2f}x")

    # The claim in the README is only worth making if the numbers back it.
    slower = [name for name, py, rs in rows if rs >= py]
    if slower:
        print(f"\n  NOTE: rust is not faster for: {', '.join(slower)}")


# --- 2. selection quality ---------------------------------------------------


def bench_selection(scale: int) -> None:
    """Does optimised selection actually buy more distinct facts than top-k?"""
    print("\n## Memory selection: optimised vs naive top-k\n")
    with BeliefStore() as store:
        # Deliberately redundant: five ways of saying each of four things.
        facts = [
            "Qubits can hold a superposition of states",
            "Rust guarantees memory safety without a garbage collector",
            "Transformers use self attention over tokens",
            "Indexes speed up database lookups considerably",
        ]
        beliefs = []
        for i in range(scale):
            fact = facts[i % len(facts)]
            beliefs.append(
                Belief(statement=f"{fact} ({'also ' * (i // len(facts))}restated).", confidence=0.9)
            )
        store.save_all(beliefs)

        evidence = EvidenceEngine(store)
        query = "qubits rust transformers indexes"
        budget = 24

        naive = []
        spent = 0
        for belief, _ in evidence.retrieve(query, limit=scale):
            cost = OptimizationEngine.cost(belief)
            if spent + cost > budget:
                continue
            naive.append(belief)
            spent += cost

        optimised = OptimizationEngine(evidence).select(query, budget=budget, pool=12)

        print(f"  {'strategy':<12} {'picked':>7} {'tokens':>7} {'distinct topics':>16}")
        for name, chosen in (("top-k", naive), ("optimised", optimised)):
            topics = {distinct_topic(b.statement, facts) for b in chosen}
            spent = sum(OptimizationEngine.cost(b) for b in chosen)
            print(f"  {name:<12} {len(chosen):>7} {spent:>7} {len(topics):>16}")
        print("\n  Distinct topics is the number that matters: the same budget should")
        print("  buy different facts, not the same fact restated.")


def distinct_topic(statement: str, facts: list[str]) -> int:
    for i, fact in enumerate(facts):
        if statement.startswith(fact[:20]):
            return i
    return -1


# --- 3. solver backends -----------------------------------------------------


def bench_solvers() -> None:
    """Annealing is only interesting if it matches exact search at lower cost."""
    print("\n## Solvers: exact vs greedy vs annealing\n")
    sizes = [8, 12, 16]
    print(f"  {'vars':>4} {'exact':>12} {'greedy':>12} {'anneal':>12}   {'both optimal?':>14}")
    for n in sizes:
        relevances = [0.5 + (i % 7) / 10 for i in range(n)]
        statements = [f"claim {i % 5} variant {i}" for i in range(n)]
        sims = {
            (i, j): jaccard(statements[i], statements[j]) for i in range(n) for j in range(i + 1, n)
        }
        qubo = build_selection_qubo(relevances, sims, redundancy=1.2)
        feasible = budget_constraint([3] * n, budget=12)

        exact_ms = timed(lambda q=qubo, f=feasible: solve_exhaustive(q, f), repeats=3)
        greedy_ms = timed(lambda q=qubo, f=feasible: solve_greedy(q, f), repeats=3)
        anneal_ms = timed(lambda q=qubo, f=feasible: simulated_annealing(q, f), repeats=3)

        best = qubo.energy(solve_exhaustive(qubo, feasible))
        greedy_e = qubo.energy(solve_greedy(qubo, feasible))
        anneal_e = qubo.energy(simulated_annealing(qubo, feasible))
        agree = (
            "yes"
            if abs(greedy_e - best) < 1e-9 and abs(anneal_e - best) < 1e-9
            else f"greedy{greedy_e - best:+.3f} anneal{anneal_e - best:+.3f}"
        )
        print(
            f"  {n:>4} {exact_ms:>10.2f}ms {greedy_ms:>10.2f}ms {anneal_ms:>10.2f}ms   {agree:>14}"
        )
    print("\n  Exact cost grows as 2^n; the others do not. Equal energy means the")
    print("  cheap solvers found the same answer, which is the whole bet behind")
    print("  handing this QUBO to a quantum backend later.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=400, help="synthetic beliefs to generate")
    args = parser.parse_args()

    print(f"CME benchmarks — scale={args.scale}, native graph={'yes' if AVAILABLE else 'no'}")
    bench_graph(args.scale)
    bench_selection(args.scale)
    bench_solvers()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
