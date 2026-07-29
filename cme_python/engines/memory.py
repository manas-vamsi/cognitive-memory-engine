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
from datetime import UTC, datetime

from pydantic import BaseModel

from cme_python.models import DEAD_BELOW, Belief, Change, MemoryTier, Revision
from cme_python.store import BeliefStore

DEFAULT_HALF_LIFE_DAYS = 180.0
"""Six months to halve an untouched belief.

Long on purpose. Memory that fades in weeks is not memory, and the point of
this engine is accumulation; the aim is only that a note nobody has revisited
in years stops outranking last week's.
"""

DECAY_FLOOR = 0.05
"""Age alone can never push a belief below this.

Falling to zero would let time delete a belief through `prune`, and "nobody
mentioned it lately" is not evidence of falsity. Disproving is what evidence is
for.
"""


class MemoryStats(BaseModel):
    total: int
    """Beliefs in play — what recall can reach."""
    by_tier: dict[str, int]
    retired: int = 0
    """Superseded beliefs: still stored, still auditable, never returned."""

    @property
    def stored(self) -> int:
        """Rows in the registry, retired ones included."""
        return self.total + self.retired


class MemoryView(BaseModel):
    """A read-only slice of memory: one tier, optionally one owner."""

    tier: MemoryTier | None = None
    scope: str | None = None

    def matches(self, belief: Belief) -> bool:
        return self.allows(belief.tier, belief.scope)

    def allows(self, tier: MemoryTier | str, scope: str | None) -> bool:
        """The same decision from tier and scope alone.

        Retrieval checks membership before it loads anything, so the filter has
        to work without a `Belief` in hand — otherwise scoping would force a
        fetch of every candidate just to reject most of them.
        """
        if self.tier is not None and str(tier) != str(self.tier):
            return False
        return not (self.scope is not None and scope != self.scope)


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
        """File beliefs into a tier and persist them, in one transaction."""
        filed = []
        for belief in beliefs:
            belief.tier = tier
            belief.scope = scope
            filed.append(belief)
        self.store.save_all(filed)
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
        # Retired beliefs included: someone exercising deletion means all of it.
        doomed = self.store.all(tier=tier, scope=scope, include_retired=True)
        for belief in doomed:
            self.store.delete(belief.id)
        return len(doomed)

    def supersede(self, old_id: str, new_id: str) -> Belief | None:
        """Retire one belief in favour of another. Returns the retired belief.

        The replacement is not touched: it stands on its own evidence, and the
        back-link is enough to find it from the belief it replaced.
        """
        old, new = self.store.get(old_id), self.store.get(new_id)
        if old is None or new is None:
            return None
        return self.store.save(old.supersede(new))

    def timeline(self, belief_id: str) -> list[Revision]:
        """How a belief's confidence got to where it is, oldest first."""
        belief = self.store.get(belief_id)
        return belief.history if belief else []

    def prune(self, threshold: float = DEAD_BELOW) -> int:
        """Clear out beliefs that have been disproven into irrelevance."""
        return self.store.prune_dead(threshold)

    def decay(
        self,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
        tier: MemoryTier | None = None,
        scope: str | None = None,
        floor: float = DECAY_FLOOR,
    ) -> dict[str, float]:
        """Age unreinforced beliefs, halving confidence every `half_life_days`.

        **Opt-in.** Nothing calls this on your behalf, because quietly weakening
        somebody's memories is not a sensible default — a fact recorded once and
        never mentioned again may be no less true.

        It exists because a registry only grows. Retrieval cost scales with
        size, and a year-old registry of notes nobody has touched since is both
        slower and less useful than the recent slice. Decay gives `prune` a
        principled way to identify what has fallen out of use, rather than
        deleting by age directly — which would throw away a belief that is
        simply settled.

        Each run ages only the span since the last one, and stamps
        `confidence_at` as it goes. Halving is memoryless — two runs over
        consecutive spans land exactly where one run over the whole span would —
        so calling this hourly and calling it yearly agree, and evidence resets
        the clock by stamping the same field.

        Confidence never falls below `floor` from age alone: only contradicting
        *evidence* should be able to disprove a belief, and that path is
        `add_evidence`. Age makes a belief quieter, not wrong.

        Returns the beliefs that moved, mapped to their new confidence.
        """
        moment = now or datetime.now(UTC)
        moved: dict[str, float] = {}
        for belief in self.store.all(tier=tier, scope=scope):
            days = (moment - belief.confidence_at).total_seconds() / 86_400
            if days <= 0:
                continue
            decayed = max(round(belief.confidence * 0.5 ** (days / half_life_days), 6), floor)
            if decayed >= belief.confidence:
                continue
            belief.confidence = decayed
            belief.record(Change.DECAYED, at=moment)
            self.store.save(belief)
            moved[belief.id] = decayed
        return moved

    def stats(self) -> MemoryStats:
        """What is remembered. `total` counts what recall can reach.

        A retired belief is still stored and still auditable, but reporting it
        as remembered would promise memory that no query can return.
        """
        by_tier = self.store.count_by_tier()
        return MemoryStats(
            total=sum(by_tier.values()),
            by_tier=by_tier,
            retired=self.store.count_retired(),
        )
