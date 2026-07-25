"""Memory Engine — multi-tier persistent memory.

The registry stores beliefs; this decides *which* memory a question is allowed
to be answered from. User, scientific, organizational and project memory are
kept apart on purpose, and `scope` separates owners inside a tier.

Keeping them apart is a correctness property, not filing tidiness: a clinical
assistant must not answer from another patient's history, and one team's
convention is not a universal fact.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from cme_python.models import DEAD_BELOW, Belief, MemoryTier
from cme_python.store import BeliefStore


class MemoryStats(BaseModel):
    total: int
    by_tier: dict[str, int]


class MemoryView(BaseModel):
    """A read-only slice of memory: one tier, optionally one owner."""

    tier: MemoryTier | None = None
    scope: str | None = None

    def matches(self, belief: Belief) -> bool:
        if self.tier is not None and belief.tier != self.tier:
            return False
        return not (self.scope is not None and belief.scope != self.scope)


class MemoryEngine:
    """Accumulates knowledge over time, partitioned into tiers."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store

    def remember(
        self,
        beliefs: Iterable[Belief],
        *,
        tier: MemoryTier = MemoryTier.GENERAL,
        scope: str | None = None,
    ) -> list[Belief]:
        """File beliefs into a tier and persist them."""
        filed = []
        for belief in beliefs:
            belief.tier = tier
            belief.scope = scope
            filed.append(self.store.save(belief))
        return filed

    def recall(
        self,
        *,
        tier: MemoryTier | None = None,
        scope: str | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[Belief]:
        """Everything remembered in one slice of memory, strongest first."""
        return self.store.all(min_confidence=min_confidence, tier=tier, scope=scope, limit=limit)

    def view(self, tier: MemoryTier | None = None, scope: str | None = None) -> MemoryView:
        """A predicate other engines can filter their candidates through."""
        return MemoryView(tier=tier, scope=scope)

    def forget(self, *, tier: MemoryTier, scope: str | None = None) -> int:
        """Drop a whole tier — a user exercising deletion, a project closing.

        Returns the number of beliefs removed.
        """
        doomed = self.store.all(tier=tier, scope=scope)
        for belief in doomed:
            self.store.delete(belief.id)
        return len(doomed)

    def prune(self, threshold: float = DEAD_BELOW) -> int:
        """Clear out beliefs that have been disproven into irrelevance."""
        return self.store.prune_dead(threshold)

    def stats(self) -> MemoryStats:
        return MemoryStats(total=len(self.store), by_tier=self.store.count_by_tier())
