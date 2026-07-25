"""Core CME data model: the Belief and the Evidence that supports it.

A Belief is the atomic unit of cognition — a structured, living claim rather than
a slice of text. It carries its own confidence, the evidence behind it, and the
links that place it in the knowledge graph.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex


class SourceKind(StrEnum):
    """Structural origin of a belief or a piece of evidence."""

    OFFICIAL_DOCS = "official_docs"
    RESEARCH_PAPER = "research_paper"
    BOOK = "book"
    CODE = "code"
    CONVERSATION = "conversation"
    WEB = "web"
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """A verifiable source snippet that supports or contradicts a belief."""

    id: str = Field(default_factory=_uid)
    snippet: str
    source: SourceKind = SourceKind.UNKNOWN
    locator: str | None = None
    """Where the snippet came from: a URL, DOI, file path, or citation."""
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    """How much this snippet moves confidence. 0 = ignorable, 1 = decisive."""
    supports: bool = True
    """False marks contradicting evidence, which pushes confidence down."""
    created_at: datetime = Field(default_factory=_now)


class Belief(BaseModel):
    """A structured claim with tracked confidence, evidence, and connections."""

    id: str = Field(default_factory=_uid)
    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    connections: set[str] = Field(default_factory=set)
    """Concept labels or belief ids this belief is semantically linked to."""
    source: SourceKind = SourceKind.UNKNOWN
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("statement")
    @classmethod
    def _statement_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("a belief needs a statement")
        return v

    # --- lifecycle: beliefs are alive -------------------------------------

    def add_evidence(self, ev: Evidence) -> Belief:
        """Attach evidence and let it move confidence. Returns self."""
        self.evidence.append(ev)
        self.confidence = _shift(self.confidence, ev.strength, ev.supports)
        self.updated_at = _now()
        return self

    def connect(self, *labels: str) -> Belief:
        self.connections.update(l for l in labels if l)
        self.updated_at = _now()
        return self

    @property
    def is_dead(self) -> bool:
        """Thoroughly disproven — drops out of active reasoning."""
        return self.confidence <= DEAD_BELOW

    def merge(self, other: Belief) -> Belief:
        """Fold a duplicate belief into this one. Returns self."""
        if other.id == self.id:
            return self
        seen = {e.id for e in self.evidence}
        self.evidence.extend(e for e in other.evidence if e.id not in seen)
        self.connections |= other.connections
        # Evidence-weighted average: the better-supported belief dominates.
        w_self, w_other = len(self.evidence) or 1, len(other.evidence) or 1
        self.confidence = (self.confidence * w_self + other.confidence * w_other) / (
            w_self + w_other
        )
        self.updated_at = _now()
        return self


DEAD_BELOW = 0.02
_CLAMP = 0.001  # keeps log-odds finite at the extremes


def _shift(confidence: float, strength: float, supports: bool) -> float:
    """Bayesian-style confidence update in log-odds space.

    Log-odds keeps updates commutative and bounded: repeated evidence pushes
    confidence toward 1 (or 0) without ever overshooting, and the same set of
    evidence lands on the same confidence whatever order it arrives in.
    """
    p = min(max(confidence, _CLAMP), 1 - _CLAMP)
    logit = math.log(p / (1 - p))
    # strength 0..1 maps to a nudge of 0..~3 log-odds (0.5 -> ~1.1, decisive).
    nudge = -math.log(1 - min(strength, 0.95))
    logit += nudge if supports else -nudge
    return round(1 / (1 + math.exp(-logit)), 6)
