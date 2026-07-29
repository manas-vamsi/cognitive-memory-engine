"""Self-check for the Belief lifecycle. Run: python tests/python_tests/test_models.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.models import HISTORY_LIMIT, Belief, Change, Evidence, SourceKind


def _belief() -> Belief:
    return Belief(statement="Python is commonly used in Machine Learning.")


def test_blank_statement_rejected():
    with pytest.raises(ValueError):
        Belief(statement="   ")


def test_supporting_evidence_raises_confidence():
    b = _belief()
    before = b.confidence
    b.add_evidence(Evidence(snippet="PyTorch docs", source=SourceKind.OFFICIAL_DOCS, strength=0.8))
    assert b.confidence > before
    assert len(b.evidence) == 1


def test_contradicting_evidence_lowers_confidence_and_can_kill():
    b = _belief()
    for _ in range(5):
        b.add_evidence(Evidence(snippet="refuted", strength=0.9, supports=False))
    assert b.confidence < 0.5
    assert b.is_dead


def test_confidence_stays_in_range_and_is_order_independent():
    strong = Evidence(snippet="a", strength=0.9)
    weak = Evidence(snippet="b", strength=0.3, supports=False)
    forward = _belief().add_evidence(strong).add_evidence(weak).confidence
    reverse = _belief().add_evidence(weak).add_evidence(strong).confidence
    assert forward == pytest.approx(reverse)
    assert 0.0 <= forward <= 1.0


def test_merge_folds_evidence_and_connections_once():
    a = _belief().connect("Programming")
    b = Belief(statement="Python is used in ML.", confidence=0.9).connect("Machine Learning")
    b.add_evidence(Evidence(snippet="TensorFlow docs", strength=0.7))
    a.merge(b)
    assert a.connections == {"Programming", "Machine Learning"}
    assert len(a.evidence) == 1
    a.merge(b)  # idempotent on evidence
    assert len(a.evidence) == 1


# --- the timeline -----------------------------------------------------------


def test_a_belief_opens_with_its_own_creation():
    b = _belief()
    assert [r.cause for r in b.history] == [Change.CREATED]
    assert b.history[0].confidence == b.confidence


def test_a_belief_stored_before_timelines_is_dated_from_its_creation():
    raw = _belief().model_dump(mode="json")
    del raw["history"]
    recovered = Belief.model_validate(raw)
    assert [r.cause for r in recovered.history] == [Change.CREATED]
    assert recovered.history[0].at == recovered.created_at


def test_the_timeline_records_what_moved_the_number():
    b = _belief()
    b.add_evidence(Evidence(snippet="PyTorch docs", locator="pytorch.org", strength=0.8))
    b.merge(Belief(statement="Python is used in ML.", confidence=0.9))
    assert [r.cause for r in b.history] == [Change.CREATED, Change.EVIDENCE, Change.MERGED]
    assert [r.confidence for r in b.history] == pytest.approx(
        [0.5, b.history[1].confidence, b.confidence]
    )
    assert b.history[1].note == "pytorch.org"


def test_last_verified_ignores_changes_nobody_checked():
    """Decay moves a belief without anyone asking whether it is still true."""
    b = _belief()
    b.add_evidence(Evidence(snippet="PyTorch docs", strength=0.8))
    checked = b.last_verified
    b.record(Change.DECAYED)
    assert b.last_verified == checked
    assert b.confidence_at > checked


def test_a_split_carries_the_parents_past_into_each_part():
    """Each part inherits evidence, so it must inherit why that evidence exists."""
    b = Belief(statement="Rust is fast and has no garbage collector.")
    b.add_evidence(Evidence(snippet="benchmark", strength=0.7))
    parts = b.split("Rust is fast.", "Rust has no garbage collector.")
    for part in parts:
        assert [r.cause for r in part.history] == [Change.CREATED, Change.EVIDENCE, Change.SPLIT]
        assert part.history[-1].note == b.id


def test_supersede_retires_without_disproving():
    """A repealed rule was not false when it was written."""
    old = Belief(statement="The rate is 4%.", confidence=0.9)
    new = Belief(statement="The rate is 5%.", confidence=0.9)
    old.supersede(new)
    assert old.superseded_by == new.id
    assert old.confidence == 0.9  # overtaken, not refuted
    assert old.history[-1].cause == Change.SUPERSEDED


def test_a_belief_cannot_supersede_itself():
    b = _belief()
    assert b.supersede(b).superseded_by is None


def test_the_timeline_is_capped():
    """History rides in the belief's row, so it cannot grow without limit."""
    b = _belief()
    for _ in range(HISTORY_LIMIT + 10):
        b.record(Change.DECAYED)
    assert len(b.history) == HISTORY_LIMIT
    assert b.history[0].cause == Change.DECAYED  # oldest dropped, newest kept


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
