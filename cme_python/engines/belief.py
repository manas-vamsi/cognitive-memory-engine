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

_WORD = re.compile(r"[a-zA-Z0-9']+")
_ENDS_A_SENTENCE = re.compile(r"[.!]$")


def _long_enough(sentence: str) -> bool:
    """Does this have enough words to be a claim rather than a heading?

    Whitespace counting made "Rust is memory-safe." three words and dropped it,
    while "Rust is memory safe." was four and survived — the same claim, kept or
    lost on a hyphen. Compounds are common in exactly the technical prose this
    reads, and a silently discarded claim is the worst failure the extractor
    has: nothing downstream can tell a fact that was never learnt from one that
    was never true.

    Counting hyphenated parts everywhere would fix that and break the other
    half, because it also inflates the fragments this is meant to reject —
    "Well-known best-practice guide" becomes five words and reads as a claim.

    So the looser count applies only to text that ends like a sentence.
    Headings and list items are not punctuated; claims are. That keeps every
    fragment currently rejected rejected, and only ever admits more.

    Three-word sentences are the other half of the same problem. "Rust is
    fast." could not be extracted from a document, yet arrived in the registry
    happily as a split of "Rust is fast and has no garbage collector" — the
    same claim, present or absent depending on how the author phrased the
    sentence around it. `split_claims` already accepts it at three words and
    says so; this is where the two disagreed.

    At three words the length has stopped carrying any signal, so the verb does
    it instead — the same test `split_claims` applies to its parts. "Rust is
    fast." has one; "See figure 3.", "Install the package." and "Table of
    contents." do not, and stay out.
    """
    if not _ENDS_A_SENTENCE.search(sentence):
        return len(sentence.split()) >= MIN_WORDS
    words = _WORD.findall(sentence)
    if len(words) >= MIN_WORDS:
        return True
    return len(words) >= MIN_CLAIM_WORDS and any(w.lower() in _VERBS for w in words)


MIN_CLAIM_WORDS = 3
"""Lower than MIN_WORDS, and deliberately so.

MIN_WORDS decides whether a line of a document is worth extracting at all.
This decides whether a *part* of an already-accepted claim stands alone, and
"Rust is fast." is a complete claim at three words. Reusing the extraction
threshold here would refuse to split exactly the short, sharp claims that are
easiest to verify.
"""

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
        if s.endswith("?") or _NOISE.match(s) or not _long_enough(s):
            continue
        claims.append(s)
    return claims


_JOIN = re.compile(r"\s*;\s+|,?\s+\band\b\s+|,?\s+\bbut\b\s+", re.I)
_VERBS = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "may",
        "might",
        "must",
        "should",
        "does",
        "do",
        "did",
        "gives",
        "give",
        "provides",
        "provide",
        "uses",
        "use",
        "requires",
        "require",
        "supports",
        "support",
        "allows",
        "allow",
        "needs",
        "need",
        "holds",
        "hold",
        "runs",
        "run",
    ]
)
"""Deliberately a closed list of auxiliaries and common verbs.

Guessing at verbs by suffix is what breaks this: "Rust gives zero-cost
abstractions and guaranteed thread safety" would split on "guaranteed" and
produce the nonsense "Rust guaranteed thread safety". A short explicit list
splits less and never mangles.
"""


def split_claims(statement: str) -> list[str]:
    """The independent claims inside a statement, or `[]` if there is only one.

    Conservative on purpose: a missed split leaves a belief slightly coarse,
    while a wrong split invents a claim nobody made. Every part must contain a
    recognised verb, and a part that starts with one inherits the subject of the
    part before it — otherwise "Rust is fast and has no GC" would yield the
    fragment "has no GC".

    ponytail: closed verb list and a regex, no parser. An LLM extractor or a
    dependency parse is the upgrade; this handles the common conjunction.
    """
    body = statement.rstrip(".!?")
    pieces = [p.strip() for p in _JOIN.split(body) if p.strip()]
    if len(pieces) < 2:
        return []

    subject = _subject_of(pieces[0])
    claims = []
    for piece in pieces:
        words = piece.split()
        if not any(w.lower().strip(",") in _VERBS for w in words):
            return []  # a part with no verb is a phrase, not a claim
        if words[0].lower() in _VERBS and subject:
            piece = f"{subject} {piece}"
        if len(piece.split()) < MIN_CLAIM_WORDS:
            return []
        claims.append(piece[0].upper() + piece[1:] + ".")
    return claims


def _subject_of(piece: str) -> str:
    """Everything before the first verb — carried onto parts that lack one."""
    words = piece.split()
    for i, word in enumerate(words):
        if word.lower().strip(",") in _VERBS:
            return " ".join(words[:i])
    return ""


def normalise(statement: str) -> str:
    """Comparison key for duplicate detection — case, punctuation, spacing ignored."""
    return " ".join(_NORMALISE.sub(" ", statement.lower()).split())


EXTRACT_SYSTEM = (
    "You extract factual claims from text for a knowledge base. "
    "Return one claim per line and nothing else — no numbering, no commentary. "
    "Each claim must be a complete standalone sentence: resolve pronouns and "
    "references so it still means the same thing read on its own. "
    "Split a sentence carrying two claims into two lines. "
    "Copy the wording of the source as closely as you can. "
    "Use only what the text states. Never add, infer, or complete a fact from "
    "your own knowledge. Skip questions, headings, instructions and opinions."
)

GROUNDING_AT = 0.75
"""Share of a claim's words that must appear in the source text.

The extractor is the one place a model writes directly into memory, and a
fabricated belief is far worse than a missed one: it arrives with a source
attached, gets its confidence, and is indistinguishable from a fact the
document actually contained.

So every returned claim is checked back against the text. Not exact matching —
the model is asked to resolve pronouns and split sentences, so some rewording is
the point — but a claim mostly built of words the document never used did not
come from the document. Measured on a two-sentence source, faithful claims and
reworded ones score 0.80 to 1.00 while invented ones score below 0.30, so the
bar sits in a wide gap rather than on a cliff edge.

It catches invented *vocabulary*, not invented *composition*. "Rust guarantees
a garbage collector" is built entirely from the words of a document saying the
opposite, and passes. Closing that needs entailment against the source rather
than word counting — the same model class `NLIDetector` already loads, applied
to a different question, and a larger job than this. Until then the rule-based
extractor remains the default, and this is the reason.
"""


class LLMExtractor:
    """Claim extraction by a model rather than by regexes.

        BeliefEngine(store, extractor=LLMExtractor(build_client("claude")))

    The rule-based extractor reads punctuation and a closed verb list, so it
    misses anything phrased unusually and cannot resolve "it" back to whatever
    the paragraph was about. A model does both.

    What a model also does is invent. Every claim it returns is checked back
    against the source, and one built largely of words the document never used
    is dropped — the sole job here is turning a document into things the engine
    will assert, and a plausible sentence is not evidence of anything.

    Takes any object with `complete()`, so it needs no import from the client
    package and no vendor knowledge.
    """

    def __init__(self, client, *, fallback: Extractor = rule_based_extract) -> None:
        self.client = client
        self.fallback = fallback

    def __call__(self, text: str) -> list[str]:
        try:
            reply = self.client.complete(text, system=EXTRACT_SYSTEM)
        except Exception:
            # A connector failing must not silently empty a document. The rule
            # based extractor is worse than the model and far better than
            # deciding the text contained no claims at all.
            return self.fallback(text)
        return [c for c in _claims_from(reply) if _grounded_in(c, text)]


def _claims_from(reply: str) -> list[str]:
    """One claim per line, with any list markers the model added stripped off."""
    claims = []
    for line in reply.splitlines():
        line = _MARKER.sub("", line.strip()).strip()
        if line and not _NOISE.match(line):
            claims.append(line if line[-1] in ".!" else f"{line}.")
    return claims


def _grounded_in(claim: str, text: str) -> bool:
    """Is this claim built from the document, or from the model?"""
    source = {w.lower() for w in _WORD.findall(text)}
    words = [w.lower() for w in _WORD.findall(claim)]
    if not words:
        return False
    return sum(w in source for w in words) / len(words) >= GROUNDING_AT


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
        beliefs = list(filed.values())
        self.store.save_all(beliefs)
        return beliefs

    def split(self, belief: Belief) -> list[Belief]:
        """Break a multi-claim belief into one belief per claim, in the registry.

        Returns the beliefs now on record — the parts if it split, otherwise the
        original untouched. The original is removed only when it is genuinely
        replaced, so a no-op split cannot lose a belief.
        """
        claims = split_claims(belief.statement)
        if not claims:
            return [belief]
        parts = belief.split(*claims)
        self.store.save_all(parts)
        self.store.delete(belief.id)
        return parts

    def split_all(self) -> list[Belief]:
        """Sweep the registry, splitting every belief that carries two claims."""
        changed = []
        for belief in self.store.all():
            parts = self.split(belief)
            if parts != [belief]:
                changed.extend(parts)
        return changed

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
