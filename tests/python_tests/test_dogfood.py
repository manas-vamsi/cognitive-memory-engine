"""The extractor, run over this repository's own documentation.

Every fixture in the rest of the suite is a sample of what I expected a
document to look like. This one is not written for the test at all, which is
the entire point: run over the README, the extractor filed the project's title
as a fact, truncated every list item that wrapped, and put markdown markup into
statements meant to be read back to a model. None of that was visible in a
hand-written fixture, because a hand-written fixture contains the cases its
author already has in mind.

Run: python tests/python_tests/test_dogfood.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.belief import rule_based_extract

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))

MARKUP = ("**", "`", "](", "|", "```")

UNPUNCTUATED_BUDGET = 0.05
"""Share of claims allowed to end without a full stop.

Not zero, and deliberately. What is left at this size is LaTeX, a shell command
and a citation fragment — the tail of a rule-based extractor reading prose it
was never going to parse perfectly, and the reason the LLM extractor exists. A
budget says "this may not get worse" without pretending the rules are finished.
"""


@pytest.fixture(scope="module", params=[p.name for p in DOCS])
def document(request):
    path = next(p for p in DOCS if p.name == request.param)
    text = path.read_text(encoding="utf-8")
    return path, text, rule_based_extract(text)


def test_the_documentation_still_yields_claims(document):
    """A filter that removed everything would pass every other test here."""
    _, _, claims = document
    assert len(claims) > 10


def test_no_markdown_markup_reaches_a_statement(document):
    """A statement is read back to a model in `as_prompt`, markup and all."""
    _, _, claims = document
    assert [c for c in claims if any(t in c for t in MARKUP)] == []


def test_no_heading_is_filed_as_a_fact(document):
    """The README's own title was a belief: "Cognitive Memory Engine (CME)"."""
    _, text, claims = document
    headings = {line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")}
    assert [c for c in claims if c.rstrip(".") in headings] == []


def test_nothing_that_merely_introduces_a_list_is_a_claim(document):
    _, _, claims = document
    assert [c for c in claims if c.endswith(":")] == []


def test_the_unparseable_tail_stays_small(document):
    """Prose the rules cannot read is a budget, not a bug — but it may not grow."""
    _, _, claims = document
    loose = [c for c in claims if c[-1] not in ".!"]
    assert len(loose) <= max(UNPUNCTUATED_BUDGET * len(claims), 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
