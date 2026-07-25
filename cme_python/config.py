"""Global settings, read from the environment.

ponytail: stdlib `os.environ` with defaults, not a settings framework. Every
value here is a string or a number with a sane default; that does not need a
dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    database: str = os.environ.get("CME_DATABASE", "cme.sqlite")
    """SQLite path for the belief registry. `:memory:` for an ephemeral run."""

    solver: str = os.environ.get("CME_SOLVER", "annealing")
    """Optimization backend: exact, annealing, dwave, or qaoa."""

    context_budget: float = _float("CME_CONTEXT_BUDGET", 60)
    """Default token budget for a selected memory set."""

    min_confidence: float = _float("CME_MIN_CONFIDENCE", 0.0)
    """Beliefs below this are left out of retrieval and the graph."""


settings = Settings()
