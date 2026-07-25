"""Latency and load. The numbers that decide whether this can be deployed.

    python benchmarks/latency.py

Three questions, none of them answerable by reading the code:

  1. How long does one retrieval take, and how does it grow?
  2. What happens under concurrency?
  3. What does a single user accumulating memory for a year feel?

Timings are one machine and move between runs. The *shapes* are the point, and
two of them are uncomfortable — that is why this file exists rather than a
paragraph claiming it is fast.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cme_python.cme import CME
from cme_python.models import Belief

TOPICS = ["quantum computing", "rust memory safety", "machine learning", "databases"]


def seed(cme: CME, n: int) -> None:
    beliefs = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        beliefs.append(
            Belief(
                statement=f"Note {i}: the team decided something about {topic} in review.",
                confidence=0.5 + (i % 50) / 100,
            ).connect(topic)
        )
    cme.store.save_all(beliefs)


def best(fn, reps: int = 5) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return min(times)


def bench_growth() -> None:
    """One user, memory accumulating. The plugin case."""
    print("\n## One user, growing memory\n")
    print(f"  {'beliefs':>8} {'mode':<8} {'context()':>11} {'after 1 ingest':>16}")
    for n in (1_000, 10_000):
        for mode in ("lexical", "vector"):
            cme = CME(":memory:", retrieval=mode)
            seed(cme, n)
            cme.context("deployment decision", budget=40)
            warm = best(lambda: cme.context("architecture decision", budget=40), reps=3)
            cme.store.save(Belief(statement="One more note about testing today.", confidence=0.8))
            after = best(lambda: cme.context("testing decision", budget=40), reps=1)
            print(f"  {n:>8} {mode:<8} {warm:>10.0f}ms {after:>15.0f}ms")
            cme.close()
    print("\n  `after 1 ingest` used to be several seconds: every write rebuilt the")
    print("  whole index. It is now within noise of a warm query. What remains is")
    print("  the linear scan, which is what an ANN index (Qdrant) is for.")


def bench_concurrency() -> None:
    """Threads do not help. Worth knowing before promising throughput."""
    print("\n## Concurrent readers (1000 beliefs)\n")
    cme = CME(":memory:")
    seed(cme, 1_000)
    cme.context("warm", budget=40)

    print(f"  {'workers':>8} {'req/s':>8} {'p50':>8} {'p99':>9}")
    for workers in (1, 4, 16):
        latencies: list[float] = []
        lock = threading.Lock()

        def one(i: int) -> None:
            start = time.perf_counter()
            cme.context(f"{TOPICS[i % len(TOPICS)]} decision", budget=40)
            with lock:
                latencies.append((time.perf_counter() - start) * 1000)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(120)))
        wall = time.perf_counter() - started
        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99) - 1]
        rps = 120 / wall
        print(f"  {workers:>8} {rps:>8.1f} {statistics.median(latencies):>7.0f}ms {p99:>8.0f}ms")
    cme.close()
    print("\n  Throughput does not improve with workers and usually falls: the store")
    print("  holds one lock across every query, and the work is CPU-bound Python")
    print("  under the GIL. Scale out with processes and shared state (Postgres +")
    print("  Qdrant), not with threads in one process.")


def main() -> int:
    print("CME latency — one machine, indicative timings, reproducible shapes")
    bench_growth()
    bench_concurrency()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
