"""Self-check for the Belief Engine. Run: python tests/python_tests/test_belief_engine.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.belief import (
    SOURCE_PRIOR,
    BeliefEngine,
    normalise,
    rule_based_extract,
    split_sentences,
)
from cme_python.models import SourceKind
from cme_python.store import BeliefStore

DOC = """
# Heading

Python is commonly used in Machine Learning. Is Rust faster than Python?
Rust gives zero-cost abstractions and guaranteed thread safety.
- bullet
Too short.
"""


@pytest.fixture
def engine():
    with BeliefStore() as store:
        yield BeliefEngine(store)


def test_splitter_keeps_abbreviations_together():
    assert split_sentences("Used in ML, e.g. vision. Rust is fast.") == [
        "Used in ML, e.g. vision.",
        "Rust is fast.",
    ]


def test_extractor_drops_questions_headings_bullets_and_fragments():
    claims = rule_based_extract(DOC)
    assert claims == [
        "Python is commonly used in Machine Learning.",
        "Rust gives zero-cost abstractions and guaranteed thread safety.",
    ]


def test_a_claim_is_not_lost_to_a_hyphen():
    """Same claim, kept or dropped on punctuation nobody thinks about.

    Whitespace counting made "Rust is memory-safe." three words, under the
    four-word floor, so it vanished at ingest while the unhyphenated wording
    survived. A silently discarded claim is the worst failure the extractor has:
    nothing downstream can tell a fact that was never learnt from one that was
    never true.
    """
    assert rule_based_extract("Rust is memory-safe.") == ["Rust is memory-safe."]
    assert rule_based_extract("The state-of-the-art model wins.") == [
        "The state-of-the-art model wins."
    ]


def test_hyphens_do_not_promote_a_fragment_into_a_claim():
    """The other half: counting hyphenated parts inflates fragments too.

    "Well-known best-practice guide" reaches five words that way and would read
    as a claim, so the looser count applies only to text that ends like a
    sentence. Headings and list items are not punctuated; claims are.
    """
    assert rule_based_extract("Well-known best-practice guide") == []
    assert rule_based_extract("- well-known best-practice") == []
    assert rule_based_extract("Read-only mode") == []


def test_normalise_ignores_case_punctuation_and_spacing():
    assert normalise("Rust, which is FAST,  is used.") == normalise("rust which is fast is used")


def test_every_belief_carries_its_source_snippet_as_evidence(engine):
    beliefs = engine.extract(DOC, source=SourceKind.OFFICIAL_DOCS, locator="docs/py.md")
    assert len(beliefs) == 2
    for b in beliefs:
        assert b.confidence == SOURCE_PRIOR[SourceKind.OFFICIAL_DOCS]
        assert b.evidence[0].snippet == b.statement
        assert b.evidence[0].locator == "docs/py.md"


def test_source_sets_the_confidence_prior(engine):
    web = engine.extract("Rust gives zero-cost abstractions here.", source=SourceKind.WEB)[0]
    paper = engine.extract(
        "Rust gives zero-cost abstractions here.", source=SourceKind.RESEARCH_PAPER
    )[0]
    assert paper.confidence > web.confidence


def test_ingest_persists_and_tags_connections(engine):
    engine.ingest(DOC, source=SourceKind.BOOK, connections=["Programming"])
    stored = engine.store.all()
    assert len(stored) == 2
    assert all("Programming" in b.connections for b in stored)


def test_reingesting_merges_instead_of_duplicating(engine):
    engine.ingest(DOC, source=SourceKind.WEB, locator="a")
    engine.ingest(DOC, source=SourceKind.OFFICIAL_DOCS, locator="b")
    stored = engine.store.all()
    assert len(stored) == 2  # not 4
    assert {e.locator for e in stored[0].evidence} == {"a", "b"}


def test_duplicate_detection_survives_mid_sentence_punctuation(engine):
    text = "Rust, which is compiled, gives real thread safety."
    engine.ingest(text)
    engine.ingest(text)
    assert len(engine.store.all()) == 1


def test_a_custom_extractor_plugs_in(engine):
    engine.extract_claims = lambda text: ["One claim only."]
    assert [b.statement for b in engine.extract(DOC)] == ["One claim only."]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
