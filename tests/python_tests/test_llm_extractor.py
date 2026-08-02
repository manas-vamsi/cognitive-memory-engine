"""Self-check for LLM-backed extraction. Run: python tests/python_tests/test_llm_extractor.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.belief import BeliefEngine, LLMExtractor, rule_based_extract
from cme_python.store import BeliefStore

DOC = "Rust guarantees memory safety without a garbage collector. It was first released in 2015."


class Canned:
    """A model that says exactly what a test tells it to."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system = None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.system = system
        return self.reply


class Broken:
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("the vendor is having a day")


def _has(module: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is not None


needs_model = pytest.mark.skipif(
    not (_has("transformers") and _has("torch")),
    reason="the entailment grounder needs transformers and torch",
)


def test_one_claim_per_line():
    extract = LLMExtractor(
        Canned(
            "Rust guarantees memory safety without a garbage collector.\nRust was released in 2015."
        )
    )
    assert extract(DOC) == [
        "Rust guarantees memory safety without a garbage collector.",
        "Rust was released in 2015.",
    ]


def test_list_markers_and_blank_lines_are_stripped():
    """Models add bullets however firmly you ask them not to."""
    extract = LLMExtractor(
        Canned("- Rust guarantees memory safety.\n\n2. Rust was released in 2015.\n---")
    )
    assert extract(DOC) == ["Rust guarantees memory safety.", "Rust was released in 2015."]


def test_a_claim_the_document_does_not_support_is_dropped():
    """The one that matters. A model writing straight into memory can invent.

    A fabricated belief is worse than a missed one: it arrives with a source
    attached, takes its confidence, and reads exactly like a fact the document
    contained.
    """
    extract = LLMExtractor(
        Canned(
            "Rust guarantees memory safety without a garbage collector.\n"
            "Rust was designed by Mozilla to replace Kubernetes in datacentres."
        )
    )
    assert extract(DOC) == ["Rust guarantees memory safety without a garbage collector."]


def test_rewording_the_document_still_counts_as_grounded():
    """The model is asked to resolve pronouns, so it cannot be exact matching.

    "It was first released in 2015" has to become "Rust was first released in
    2015" to stand alone, and that rewrite must survive the check.
    """
    extract = LLMExtractor(Canned("Rust was first released in 2015."))
    assert extract(DOC) == ["Rust was first released in 2015."]


RECOMBINED = "Rust guarantees a garbage collector."
SOURCE = "Rust guarantees memory safety without a garbage collector."


def test_word_overlap_alone_cannot_see_invented_meaning():
    """Why the grounder exists, kept as the contrast to the test below.

    Every word of "Rust guarantees a garbage collector" appears in a document
    saying the opposite, so counting words cannot tell them apart.
    """
    assert LLMExtractor(Canned(RECOMBINED))(SOURCE) == [RECOMBINED]


@needs_model
def test_the_grounder_rejects_a_claim_the_source_contradicts():
    """The hole from the word-overlap check, closed.

    Same claim, same source, same perfect word overlap — and now dropped,
    because the model is asked whether the document entails it rather than
    whether it reuses its vocabulary.
    """
    from cme_python.engines.entailment import NLIGrounder

    extract = LLMExtractor(Canned(RECOMBINED), grounder=NLIGrounder())
    assert extract(SOURCE) == []


@needs_model
def test_the_grounder_keeps_what_the_source_actually_says():
    """A guard that rejects everything would pass the test above and be useless."""
    from cme_python.engines.entailment import NLIGrounder

    extract = LLMExtractor(
        Canned("Rust guarantees memory safety.\nRust was first released in 2015."),
        grounder=NLIGrounder(),
    )
    assert extract(DOC) == ["Rust guarantees memory safety.", "Rust was first released in 2015."]


@needs_model
def test_a_claim_the_source_merely_permits_is_not_believed():
    """Neutral is not support. "The document does not rule this out" is not a
    reason for a memory engine to assert something."""
    from cme_python.engines.entailment import NLIGrounder

    extract = LLMExtractor(Canned("Rust was released in 2015 by Mozilla."), grounder=NLIGrounder())
    assert extract(DOC) == []


def test_a_failing_model_falls_back_rather_than_emptying_the_document():
    """A connector having a bad minute must not read as "this text said nothing"."""
    extract = LLMExtractor(Broken())
    assert extract(DOC) == rule_based_extract(DOC)
    assert extract(DOC)  # and that is not the empty list


def test_the_extractor_is_told_not_to_invent():
    """The prompt is the first defence; the grounding check is the second."""
    model = Canned("Rust guarantees memory safety.")
    LLMExtractor(model)(DOC)
    assert "Never add, infer, or complete a fact" in model.system


def test_extracted_claims_become_beliefs_with_the_source_attached():
    """End to end through the engine the rule-based extractor normally feeds."""
    with BeliefStore() as store:
        engine = BeliefEngine(
            store, extractor=LLMExtractor(Canned("Rust guarantees memory safety."))
        )
        filed = engine.ingest(DOC, locator="the Rust book")

        assert [b.statement for b in filed] == ["Rust guarantees memory safety."]
        assert filed[0].evidence[0].locator == "the Rust book"
        assert store.get(filed[0].id) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
