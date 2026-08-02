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

    graph_backend: str = os.environ.get("CME_GRAPH", "auto")
    """`auto` uses the Rust core when built, `python` forces the reference."""

    retrieval: str = os.environ.get("CME_RETRIEVAL", "lexical")
    """`lexical` (TF-IDF) or `vector` (embedding similarity)."""

    detector: str = os.environ.get("CME_DETECTOR", "lexical")
    """`lexical` (negation parity) or `nli` (an entailment model).

    Lexical by default because it needs no model, no download and no machine
    learning runtime, and it catches the clash that matters most: the same claim
    asserted and denied. `nli` reads meaning instead of form and finds
    disagreements nobody phrased as a negation, at the cost of `transformers`
    and `torch`.
    """

    extractor: str = os.environ.get("CME_EXTRACTOR", "rule")
    """`rule` (regex and a closed verb list) or `llm` (the configured model).

    Rule-based by default: it runs offline, costs nothing, and never invents a
    claim. `llm` reads phrasing the rules miss and resolves pronouns, and needs
    `CME_LLM` set to a connector that works.
    """

    grounding: str = os.environ.get("CME_GROUNDING", "words")
    """How an extracted claim is checked against its source: `words` or `entailment`.

    Word overlap is free and catches invented vocabulary. `entailment` asks a
    model whether the document actually entails the claim, which additionally
    catches a claim built from the document's own words to say the opposite —
    at a model call per claim, and the `entailment` extra.
    """

    llm: str = os.environ.get("CME_LLM", "")
    """`claude` or `openai` to enable /ask. Empty leaves the endpoint disabled."""

    llm_model: str = os.environ.get("CME_LLM_MODEL", "")
    """Overrides the connector's default model."""


settings = Settings()
