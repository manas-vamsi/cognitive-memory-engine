"""Core CME data model: the Belief and the Evidence that supports it.

A Belief is the atomic unit of cognition — a structured, living claim rather than
a slice of text. It carries its own confidence, the evidence behind it, and the
links that place it in the knowledge graph.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


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


class MemoryTier(StrEnum):
    """Which body of memory a belief belongs to.

    The tiers are separate so recall can be scoped: a clinical assistant should
    not answer from another patient's history, and a project assistant should
    not treat one team's conventions as universal fact.
    """

    GENERAL = "general"
    USER = "user"
    SCIENTIFIC = "scientific"
    ORGANIZATIONAL = "organizational"
    PROJECT = "project"


class Change(StrEnum):
    """Why a belief's confidence moved.

    Knowing a belief sits at 0.3 says little; knowing it was born at 0.8, was
    contradicted twice and then faded is a different claim about the world than
    one that has simply never been mentioned again.
    """

    CREATED = "created"
    EVIDENCE = "evidence"
    MERGED = "merged"
    SPLIT = "split"
    DECAYED = "decayed"
    CONTRADICTED = "contradicted"
    PROPAGATED = "propagated"
    SUPERSEDED = "superseded"


class Revision(BaseModel):
    """One entry in a belief's timeline: what changed it, when, and to what."""

    at: datetime = Field(default_factory=_now)
    cause: Change
    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence *after* the change, so the list reads as a curve."""
    note: str | None = None
    """What did it — an evidence locator, or the id of the other belief."""


HISTORY_LIMIT = 100
"""Revisions kept per belief, newest wins.

ponytail: a flat cap, because history rides in the belief's JSON row and a
belief reinforced ten thousand times would otherwise make every read of it
expensive. Move to a side table if anyone needs the full audit trail.
"""


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
    tier: MemoryTier = MemoryTier.GENERAL
    scope: str | None = None
    """Owner within the tier — a user id, project name, or organisation.

    Two projects both storing "we deploy on Fridays" are different memories, not
    a contradiction; the tier says what kind of memory it is and the scope says
    whose.
    """

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    """When the row last changed. Indexes use this to detect what to reindex."""

    confidence_at: datetime = Field(default_factory=_now)
    """When the current confidence was last computed.

    Deliberately not `updated_at`. Ageing needs the age of the *number*, while
    indexes need the age of the *row* — and a rewritten statement changes the
    row without touching confidence. `decay` moves this stamp forward too:
    halving is memoryless, so ageing the gap since the last calculation is
    identical to ageing the whole span, and it cannot compound.
    """

    history: list[Revision] = Field(default_factory=list)
    """How the confidence got to where it is, oldest first.

    The point of the engine is knowledge that changes; a bare number cannot say
    whether it changed because the world did, because someone found a better
    source, or because nobody has looked in a year.
    """

    superseded_by: str | None = None
    """The belief that replaced this one, if a better-sourced claim arrived.

    Distinct from being disproven. A superseded belief was not wrong so much as
    overtaken, and its timeline is why the replacement is trusted — so it leaves
    active recall but is never deleted.
    """

    @model_validator(mode="after")
    def _open_the_timeline(self) -> Belief:
        if not self.history:
            # Dated from creation, not from now: a belief loaded out of a
            # registry written before timelines existed was not born today.
            self.history = [
                Revision(at=self.created_at, cause=Change.CREATED, confidence=self.confidence)
            ]
        return self

    @model_validator(mode="before")
    @classmethod
    def _stamp_legacy_rows(cls, data: object) -> object:
        # Rows written before this field existed would otherwise load as though
        # their confidence were calculated the moment we read them, which resets
        # the age of every belief in an old registry on first open.
        if isinstance(data, dict) and "confidence_at" not in data and "updated_at" in data:
            data = {**data, "confidence_at": data["updated_at"]}
        return data

    @field_validator("statement")
    @classmethod
    def _statement_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("a belief needs a statement")
        return v

    # --- lifecycle: beliefs are alive -------------------------------------

    def record(self, cause: Change, note: str | None = None, *, at: datetime | None = None) -> None:
        """Append to the timeline and move the confidence clock with it.

        Every path that changes confidence goes through here, so the timeline
        cannot drift out of step with the number it explains.
        """
        self.confidence_at = at or _now()
        self.updated_at = self.confidence_at
        self.history.append(
            Revision(at=self.confidence_at, cause=cause, confidence=self.confidence, note=note)
        )
        del self.history[:-HISTORY_LIMIT]

    @property
    def last_verified(self) -> datetime:
        """When evidence last spoke to this claim — not when the row last moved.

        Decay and renaming touch a belief without anyone checking whether it is
        still true. This is the honest answer to "how stale is this?".
        """
        checked = [r.at for r in self.history if r.cause in (Change.EVIDENCE, Change.MERGED)]
        return max(checked) if checked else self.created_at

    def add_evidence(self, ev: Evidence) -> Belief:
        """Attach evidence and let it move confidence. Returns self."""
        self.evidence.append(ev)
        self.confidence = _shift(self.confidence, ev.strength, ev.supports)
        self.record(Change.EVIDENCE, ev.locator or ev.snippet[:60])
        return self

    def supersede(self, replacement: Belief) -> Belief:
        """Retire this belief in favour of a better-sourced one. Returns self.

        Not a merge: merging assumes both claims are the same claim. This is for
        the case where the new belief *contradicts* the old one and wins — a
        revised figure, a repealed rule — where averaging the two would invent a
        number nobody ever claimed.
        """
        if replacement.id == self.id:
            return self
        self.superseded_by = replacement.id
        self.record(Change.SUPERSEDED, replacement.id)
        return self

    def connect(self, *labels: str) -> Belief:
        self.connections.update(label for label in labels if label)
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
        # A merge brings evidence with it, so it counts as reinforcement.
        self.record(Change.MERGED, other.id)
        return self

    def split(self, *statements: str) -> list[Belief]:
        """Break a belief carrying several claims into one belief per claim.

        The counterpart to `merge`. A statement like "Rust is fast and has no
        garbage collector" is two claims wearing one confidence score: evidence
        for the second silently props up the first, and neither can be refuted
        on its own.

        Each part inherits this belief's evidence, connections and confidence,
        because the evidence genuinely was gathered for the whole sentence. They
        diverge from the next piece of evidence onward, which is the point.
        """
        parts = [s.strip() for s in statements if s and s.strip()]
        if len(parts) < 2:
            return [self]
        children = [
            Belief(
                statement=part,
                confidence=self.confidence,
                evidence=[e.model_copy(deep=True) for e in self.evidence],
                connections=set(self.connections),
                source=self.source,
                tier=self.tier,
                scope=self.scope,
                created_at=self.created_at,
                # The parts inherit the parent's past: the evidence behind them
                # was gathered before the split, and dropping the timeline would
                # make each part look newly asserted with no support.
                history=[r.model_copy(deep=True) for r in self.history],
            )
            for part in parts
        ]
        for child in children:
            child.record(Change.SPLIT, self.id)
        return children


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
