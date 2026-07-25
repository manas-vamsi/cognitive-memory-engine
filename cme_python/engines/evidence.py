"""Evidence Engine — retrieval and grounding.

Two jobs. Retrieval: find the beliefs that actually bear on a query. Grounding:
take a block of generated text and check every claim in it against the
registry, so an ungrounded sentence is caught instead of shipped.

Scoring is pluggable. The default is lexical TF-IDF over belief statements and
their evidence snippets — deterministic, offline, no server. A vector store
(Qdrant, pgvector) drops in behind the same `Retriever` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel

from cme_python.engines.belief import split_sentences
from cme_python.models import Belief, Evidence
from cme_python.store import BeliefStore

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "can",
        "could",
        "will",
        "would",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "so",
        "such",
    ]
)

SUPPORTED_AT = 0.15
"""Relevance below this and we call the claim ungrounded."""

COVERAGE_AT = 0.5
"""Fraction of a claim's own words the matching belief must actually contain.

Relevance alone is topical, not evidential: "Qubits are powered by steam"
scores highly against any belief about qubits, because they share the subject.
Requiring the belief to cover most of the claim's words is a cheap stand-in for
entailment — it rejects a sentence that merely talks about the right thing.
Calibrated so a close paraphrase still passes; a real entailment model is the
upgrade, and this is the knob to retune when one lands.
"""

Retriever = Callable[[str, int], list[tuple[Belief, float]]]


def tokenise(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


class Justification(BaseModel):
    """Why the engine believes something, in a form a user can audit."""

    statement: str
    confidence: float
    supporting: list[Evidence]
    contradicting: list[Evidence]

    @property
    def verdict(self) -> str:
        if not self.supporting and not self.contradicting:
            return "unsupported"
        if self.contradicting and self.supporting:
            return "disputed"
        return "grounded" if self.supporting else "refuted"

    def explain(self) -> str:
        """One-paragraph answer to: why? where from? how certain?"""
        sources = [e.locator or str(e.source) for e in self.supporting] or ["none on record"]
        return (
            f'"{self.statement}" is {self.verdict} at {self.confidence:.0%} confidence, '
            f"from {len(self.supporting)} supporting and {len(self.contradicting)} "
            f"contradicting source(s): {', '.join(sources)}."
        )


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    relevance: float
    coverage: float = 0.0
    """How much of the claim the matched belief actually accounts for."""

    belief: Belief | None = None


class GroundingReport(BaseModel):
    """Result of checking generated text against what the engine actually knows."""

    checks: list[ClaimCheck]

    @property
    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.supported]

    @property
    def is_grounded(self) -> bool:
        return bool(self.checks) and not self.unsupported

    @property
    def score(self) -> float:
        """Fraction of claims the registry can back."""
        if not self.checks:
            return 0.0
        return round(sum(c.supported for c in self.checks) / len(self.checks), 4)


class EvidenceEngine:
    """Ranks beliefs against a query and verifies claims against the registry."""

    def __init__(self, store: BeliefStore, retriever: Retriever | None = None) -> None:
        self.store = store
        self._retriever = retriever
        self._docs: dict[str, Counter[str]] = {}
        self._idf: dict[str, float] = {}
        self._indexed = -1

    # --- lexical index -----------------------------------------------------

    def reindex(self) -> None:
        """Rebuild the term index over statements and evidence snippets.

        ponytail: full rebuild, O(corpus). Fine while the registry is small;
        switch to an incremental index or a real vector store when it isn't.
        """
        beliefs = self.store.all()
        self._docs = {
            b.id: Counter(tokenise(" ".join([b.statement, *(e.snippet for e in b.evidence)])))
            for b in beliefs
        }
        n = len(self._docs) or 1
        df: Counter[str] = Counter()
        for terms in self._docs.values():
            df.update(terms.keys())
        # Smoothed IDF with a +1 floor (scikit-learn's convention): rare terms
        # still outweigh common ones, but a term present in every belief keeps
        # a weight of 1 instead of 0 — otherwise a small or single-topic
        # registry scores every match at zero and retrieves nothing.
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._indexed = len(beliefs)

    def _fresh_index(self) -> None:
        if self._indexed != len(self.store):
            self.reindex()

    # --- retrieval ---------------------------------------------------------

    def retrieve(self, query: str, limit: int = 5) -> list[tuple[Belief, float]]:
        """Beliefs bearing on a query, best first, as (belief, relevance).

        Relevance is TF-IDF overlap scaled by the belief's own confidence — a
        perfectly matching but disbelieved statement should not win.
        """
        if self._retriever is not None:
            return self._retriever(query, limit)
        self._fresh_index()
        terms = tokenise(query)
        if not terms:
            return []
        scored: list[tuple[Belief, float]] = []
        for belief_id, doc in self._docs.items():
            length = sum(doc.values()) or 1
            overlap = sum(doc[t] * self._idf.get(t, 0.0) for t in terms)
            if overlap <= 0:
                continue
            belief = self.store.get(belief_id)
            if belief is None:  # deleted between index and read
                continue
            relevance = (overlap / math.sqrt(length)) * belief.confidence
            scored.append((belief, round(relevance, 6)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # --- justification -----------------------------------------------------

    def justify(self, belief: Belief) -> Justification:
        """Split a belief's evidence into what backs it and what fights it."""
        return Justification(
            statement=belief.statement,
            confidence=belief.confidence,
            supporting=[e for e in belief.evidence if e.supports],
            contradicting=[e for e in belief.evidence if not e.supports],
        )

    # --- grounding ---------------------------------------------------------

    def check(
        self,
        claim: str,
        *,
        threshold: float = SUPPORTED_AT,
        coverage_at: float = COVERAGE_AT,
    ) -> ClaimCheck:
        """Is this one sentence backed by something the engine knows?

        Two gates, and both must pass: the belief has to be *relevant* to the
        claim, and it has to *cover* the claim. Relevance alone lets a false
        sentence through on a shared subject word.
        """
        hits = self.retrieve(claim, limit=1)
        if not hits:
            return ClaimCheck(claim=claim, supported=False, relevance=0.0)
        belief, relevance = hits[0]
        coverage = self.coverage(claim, belief)
        supported = relevance >= threshold and coverage >= coverage_at
        return ClaimCheck(
            claim=claim,
            supported=supported,
            relevance=relevance,
            coverage=coverage,
            belief=belief if supported else None,
        )

    def coverage(self, claim: str, belief: Belief) -> float:
        """Fraction of the claim's content words the belief accounts for."""
        wanted = set(tokenise(claim))
        if not wanted:
            return 0.0
        known = self._docs.get(belief.id) or Counter(
            tokenise(" ".join([belief.statement, *(e.snippet for e in belief.evidence)]))
        )
        return round(len(wanted & set(known)) / len(wanted), 6)

    def ground(self, text: str, *, threshold: float = SUPPORTED_AT) -> GroundingReport:
        """Check generated text claim by claim against the registry.

        This is the hallucination guard: anything in `report.unsupported` is a
        sentence the engine cannot back with a stored belief.
        """
        return GroundingReport(checks=[self.check(c, threshold=threshold) for c in claims_in(text)])


def claims_in(text: str) -> list[str]:
    """Every sentence in `text` that asserts something.

    Deliberately not `rule_based_extract`: that drops fragments below a word
    count because they make poor beliefs, but for grounding a short sentence is
    still a claim, and skipping it would let it through unverified.
    """
    claims = [s for s in split_sentences(text) if not s.endswith("?") and tokenise(s)]
    return claims or ([text.strip()] if text.strip() else [])
