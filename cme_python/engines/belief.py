"""Belief Engine — turns raw text into structured beliefs.

Extraction is deliberately pluggable: the default extractor is rule-based and
runs offline, so ingestion works with no API key and tests stay deterministic.
An LLM-backed extractor drops in behind the same `Extractor` signature once the
model clients land.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from cme_python.models import Belief, Evidence, SourceKind
from cme_python.store import BeliefStore

# How much we trust a claim purely because of where it came from.
SOURCE_PRIOR: dict[SourceKind, float] = {
    SourceKind.OFFICIAL_DOCS: 0.75,
    SourceKind.RESEARCH_PAPER: 0.7,
    SourceKind.BOOK: 0.65,
    SourceKind.CODE: 0.6,
    SourceKind.WEB: 0.45,
    SourceKind.CONVERSATION: 0.4,
    SourceKind.UNKNOWN: 0.5,
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|fig|no|al|dr|mr|ms|st)\.$", re.I)
_NOISE = re.compile(r"^\s*(?:[-*#>\d.)\s]+)$")
_NORMALISE = re.compile(r"[^a-z0-9 ]")
_MARKER = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)")

MIN_WORDS = 4
"""Below this a fragment is a heading or a list bullet, not a claim."""

Extractor = Callable[[str], list[str]]


def split_blocks(text: str) -> list[str]:
    """Group lines into blocks before any sentence splitting.

    Wrapped prose lines belong to one sentence and are joined; a heading, a
    bullet, or a blank line ends the block. Without this, `# Heading` glues onto
    the first real sentence and the claim comes out mangled.
    """
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif _MARKER.match(raw):
            flush()
            blocks.append(_MARKER.sub("", raw).strip())
        else:
            current.append(line)
    flush()
    return blocks


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, block by block.

    ponytail: regex split with an abbreviation guard, not a parser. Swap in
    spaCy or a model extractor if real prose starts breaking it.
    """
    out: list[str] = []
    for block in split_blocks(text):
        first_in_block = True
        for part in _SENTENCE_END.split(block):
            part = part.strip()
            if not part:
                continue
            # "used in ML, e.g. vision." split badly — glue it back on.
            if out and not first_in_block and _ABBREV.search(out[-1]):
                out[-1] = f"{out[-1]} {part}"
            else:
                out.append(part)
                first_in_block = False
    return out


def rule_based_extract(text: str) -> list[str]:
    """Pull declarative claims out of text. Questions and fragments are dropped."""
    claims = []
    for s in split_sentences(text):
        if s.endswith("?") or _NOISE.match(s) or len(s.split()) < MIN_WORDS:
            continue
        claims.append(s)
    return claims


def normalise(statement: str) -> str:
    """Comparison key for duplicate detection — case, punctuation, spacing ignored."""
    return " ".join(_NORMALISE.sub(" ", statement.lower()).split())


class BeliefEngine:
    """Extracts beliefs from documents and files them in the registry."""

    def __init__(self, store: BeliefStore, extractor: Extractor = rule_based_extract) -> None:
        self.store = store
        self.extract_claims = extractor

    def extract(
        self,
        text: str,
        *,
        source: SourceKind = SourceKind.UNKNOWN,
        locator: str | None = None,
        connections: Iterable[str] = (),
    ) -> list[Belief]:
        """Build beliefs from text without persisting them."""
        prior = SOURCE_PRIOR[source]
        beliefs: list[Belief] = []
        for claim in self.extract_claims(text):
            b = Belief(statement=claim, confidence=prior, source=source)
            # The sentence is its own first evidence — every belief is traceable.
            b.evidence.append(Evidence(snippet=claim, source=source, locator=locator))
            b.connect(*connections)
            beliefs.append(b)
        return beliefs

    def ingest(
        self,
        text: str,
        *,
        source: SourceKind = SourceKind.UNKNOWN,
        locator: str | None = None,
        connections: Iterable[str] = (),
    ) -> list[Belief]:
        """Extract, fold into any duplicate already known, and save.

        Re-ingesting the same document reinforces existing beliefs instead of
        filling the registry with copies.
        """
        filed: dict[str, Belief] = {}
        for fresh in self.extract(text, source=source, locator=locator, connections=connections):
            key = normalise(fresh.statement)
            target = filed.get(key) or self._find_duplicate(fresh.statement, key)
            if target is None:
                filed[key] = fresh
            else:
                filed[key] = target.merge(fresh)
        for b in filed.values():
            self.store.save(b)
        return list(filed.values())

    def _find_duplicate(self, statement: str, key: str) -> Belief | None:
        """Look for an existing belief with the same normalised statement.

        The probe is a raw prefix so SQLite's LIKE can match it; the exact
        decision is made on the normalised form.

        ponytail: prefix LIKE probe over the registry — cheap and exact-match
        only. Semantic near-duplicates are the Reasoning Engine's job.
        """
        probe = statement[:40].replace("%", "").replace("_", "")
        if not probe:
            return None
        for candidate in self.store.search(probe, limit=50):
            if normalise(candidate.statement) == key:
                return candidate
        return None
