"""Contradiction detectors — how the engine decides two beliefs disagree.

The lexical detector reads negation. "Rust has a garbage collector" against
"Rust has no garbage collector" is the shape it was built for, and it catches
that with no model, no network and no dependency.

What it cannot see is a disagreement nobody phrased as a negation. "Rust is
memory-safe" and "Rust leaks memory freely" contradict each other with no
negation word between them, so parity reads both as affirmative and the pair is
never even scored. That is not a threshold to tune; the signal is absent.

An entailment model is the thing that reads meaning rather than form, so this
module is the seam it plugs into. `Detector` is the contract, and both
implementations return the same pairs-with-scores whatever they know.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from cme_python.models import Belief


class Detector(Protocol):
    """Finds beliefs that disagree.

    One method, because the interesting variation is *which* pairs get scored
    and how — not the bookkeeping around them. A detector may compare every
    pair, block by shared words, or block by embedding neighbourhood; the
    Reasoning Engine does not care and should not have to.
    """

    def clashes(
        self, beliefs: list[Belief], threshold: float
    ) -> list[tuple[Belief, Belief, float]]:
        """Contradicting pairs, each with a score in 0..1. Order is not promised."""
        ...


def get_detector(name: str = "lexical", **kwargs: object) -> Detector:
    """Look up a detector by name, the way `get_solver` looks up a solver.

    Imported lazily so choosing `lexical` never pays for the entailment stack,
    which is a machine-learning runtime CME does not depend on.
    """
    if name == "lexical":
        from cme_python.engines.reasoning import LexicalDetector  # noqa: PLC0415

        return LexicalDetector()
    if name in ("nli", "entailment"):
        return NLIDetector(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown detector {name!r}. Available: lexical, nli")


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
"""Small on purpose.

Contradiction detection runs over pairs, so model size multiplies. An xsmall
cross-encoder is a few hundred megabytes and reads a pair in milliseconds on a
CPU; the larger ones are better at the sentences this is least likely to see.
"""

NEIGHBOURS = 10
"""Pairs scored per belief.

The lexical detector can block soundly — a 0.75 word overlap is unreachable
without a shared word — but meaning has no such guarantee, so this one blocks by
embedding neighbourhood and that *is* a heuristic. A contradiction between two
beliefs that are not among each other's nearest neighbours will be missed. The
alternative is scoring every pair through a transformer, which is quadratic in
model calls rather than in set operations, and that is not a trade anyone wants.
"""

RELATED_AT = 0.25
"""Content overlap a pair needs before the model is asked about it at all.

Not an optimisation. NLI models are trained on premise-hypothesis pairs that are
already about the same thing, so "unrelated" is barely in their vocabulary and
they reach for "contradiction" instead:

    Rust is memory-safe / Rust leaks memory constantly     1.000   correct
    Rust is memory-safe / Rust prevents data races         0.011   correct
    Rust is memory-safe / Qubits hold a superposition      0.971   nonsense

Without a gate the detector reports most of a registry as self-contradictory,
confidently, and `reconcile` retires real beliefs on the strength of it.

The bar started at 0.1, which was enough to separate the three pairs above and
not enough for a registry. One shared stemmed word clears it, and on
twenty-five ordinary beliefs that was "TLS encrypts traffic between a client
and a server" against "Replication streams the log to standby servers"
(*server*), and "Cargo is the package manager for Rust" against "Docker
packages an application" (*package*) — both reported as contradictions.

Measured over that registry, the separation is wide and not where the first
guess put it:

    genuine contradictions      0.64 - 0.86   (six or seven shared words)
    unrelated, one word apiece  0.10 - 0.14

with the sparsest genuine pair seen anywhere — "Rust is memory-safe" against
"Rust leaks memory constantly", two shared words — at 0.400. So the bar sits at
0.25, between the worst true positive and the best false one, with room either
side rather than tucked against either.
"""


ENTAILED_AT = 0.5
"""How strongly the source must entail a claim for it to be believed.

For grounding extraction rather than finding clashes, and the separation is
wide. Asked whether a document entails a claim taken from it:

    faithful, and pronoun-resolved rewrites      0.83 - 0.99  entailment
    a claim recombining the document's words     0.001        (0.998 contradiction)
    a claim invented outright                    0.000        (0.996 neutral)

So the bar is only that entailment wins, not that it wins narrowly. Note this
catches the invented claim as well, which word overlap also caught, *and* the
recombined one, which it could not: "Rust guarantees a garbage collector" is
built entirely from the words of a document saying the opposite, and the model
calls it a contradiction at 0.998.
"""


@lru_cache(maxsize=4)
def _load(model_name: str):
    """The classifier for a model name, loaded once per process.

    Cached because the detector and the grounder ask for the same model, and a
    few hundred megabytes is not worth holding twice to save one dictionary.
    Loaded on first use, not on construction: wiring up a model and never
    asking it anything should cost nothing.
    """
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError:
        raise _missing() from None
    return pipeline("text-classification", model=model_name, top_k=None)


class NLIGrounder:
    """Checks a claim against the text it was supposedly taken from.

        LLMExtractor(client, grounder=NLIGrounder())

    The word-overlap check in the extractor sees invented *vocabulary*. This
    sees invented *meaning*, which is the failure that survives it: a claim
    assembled from the document's own words can say the opposite of the
    document and still score a perfect overlap.

    Asks the model whether the source entails the claim, and keeps it only if
    it does. Neutral is not good enough — "the document does not rule this out"
    is not a reason for a memory engine to assert something.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, entailed_at: float = ENTAILED_AT) -> None:
        self.model_name = model
        self.entailed_at = entailed_at

    def supports(self, source: str, claim: str) -> bool:
        premise = _passage_for(claim, source)
        scores = _scores(_load(self.model_name), premise, claim)
        return scores.get("entailment", 0.0) >= self.entailed_at


WINDOW = 2
"""Sentences of the source shown to the model as the premise.

Not a performance knob — the whole document does not work. A cross-encoder is
trained on single-sentence premises, and a longer one takes it out of
distribution, where it answers "neutral" to everything. Measured with the claim
sitting verbatim in the premise:

    1 sentence    0.995 entailment
    2 sentences   0.993
    3 sentences   0.993
    4 sentences   0.023   (0.976 neutral)

Passing a real four-paragraph document rejected 16 of 16 claims copied straight
out of it. Every test this was shipped with used a two-sentence source, so every
test passed.

Two rather than one because a claim is asked to stand alone: "It was first
released in 2015" becomes "Rust was first released in 2015", and the model
scores that 0.000 against the pronoun sentence by itself and 0.990 against the
sentence pair carrying the antecedent. Two is the smallest window that resolves
a pronoun, and three was the largest that still worked.
"""


def _passage_for(claim: str, source: str) -> str:
    """The stretch of the source a claim should be judged against.

    The sentence sharing the most words with the claim, plus the one before it
    — near enough always the sentence the claim came from, and its antecedent.
    """
    from cme_python.engines.belief import split_sentences  # noqa: PLC0415
    from cme_python.engines.reasoning import _content  # noqa: PLC0415

    sentences = split_sentences(source)
    if len(sentences) <= WINDOW:
        return source

    wanted = _content(claim)
    best = max(range(len(sentences)), key=lambda i: len(wanted & _content(sentences[i])))
    return " ".join(sentences[max(best - WINDOW + 1, 0) : best + 1])


def _missing() -> RuntimeError:
    return RuntimeError(
        "The NLI detector needs `transformers` and `torch`, which CME does not "
        "install by default. Run `pip install transformers torch`, or use the "
        "lexical detector, which needs neither."
    )


class NLIDetector:
    """Contradiction detection by natural-language inference.

    Reads a pair as premise and hypothesis and asks a model whether the second
    contradicts the first. That is what catches the disagreements the lexical
    detector cannot see, because it responds to meaning rather than to the word
    "not".

    Both directions are scored and the larger taken. Entailment models are not
    symmetric — "Rust leaks memory" as premise against "Rust is memory-safe" as
    hypothesis is a different question from the reverse — and a contradiction
    that only one ordering notices is still a contradiction.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        neighbours: int = NEIGHBOURS,
        related_at: float = RELATED_AT,
        embed=None,
    ) -> None:
        self.model_name = model
        self.neighbours = neighbours
        self.related_at = related_at
        self._embed = embed
        self._pipe = None

    def _classifier(self):
        return _load(self.model_name)

    def _candidates(self, beliefs: list[Belief]) -> set[tuple[int, int]]:
        """Index pairs worth asking the model about.

        Embedding neighbourhood, falling back to every pair when there is no
        retriever to ask — which is correct for a handful of beliefs and
        unaffordable for a registry, so a caller running this at size should
        pass one.
        """
        if self._embed is None or len(beliefs) <= self.neighbours + 1:
            return {(i, j) for i in range(len(beliefs)) for j in range(i + 1, len(beliefs))}

        vectors = [self._embed(b.statement) for b in beliefs]
        pairs: set[tuple[int, int]] = set()
        for i, vec in enumerate(vectors):
            scored = sorted(
                ((_cosine(vec, other), j) for j, other in enumerate(vectors) if j != i),
                reverse=True,
            )
            for _, j in scored[: self.neighbours]:
                pairs.add((min(i, j), max(i, j)))
        return pairs

    def clashes(
        self, beliefs: list[Belief], threshold: float
    ) -> list[tuple[Belief, Belief, float]]:
        from cme_python.engines.reasoning import _content  # noqa: PLC0415

        content = [_content(b.statement) for b in beliefs]
        pairs = [
            (i, j)
            for i, j in sorted(self._candidates(beliefs))
            if _overlap(content[i], content[j]) >= self.related_at
        ]
        if not pairs:
            return []

        classify = self._classifier()
        found = []
        for i, j in pairs:
            a, b = beliefs[i], beliefs[j]
            score = max(
                _contradiction_score(classify, a.statement, b.statement),
                _contradiction_score(classify, b.statement, a.statement),
            )
            if score >= threshold:
                found.append((a, b, round(score, 6)))
        return found


def _scores(classify, premise: str, hypothesis: str) -> dict[str, float]:
    """Every label the model assigns the pair, lowercased."""
    scores = classify({"text": premise, "text_pair": hypothesis})
    # `top_k=None` returns every label; the nesting depends on the pipeline
    # version, so unwrap one level if it came back wrapped.
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    return {entry["label"].lower(): float(entry["score"]) for entry in scores}


def _contradiction_score(classify, premise: str, hypothesis: str) -> float:
    """How strongly the model reads the second sentence as denying the first."""
    return _scores(classify, premise, hypothesis).get("contradiction", 0.0)


def _overlap(a: set[str], b: set[str]) -> float:
    """Jaccard over content words — are these two even about the same thing?"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine over the sparse embeddings the vector retriever already produces."""
    if not a or not b:
        return 0.0
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    norm = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return dot / norm if norm else 0.0
