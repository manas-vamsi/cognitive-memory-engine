"""Self-check for the Belief lifecycle. Run: python tests/python_tests/test_models.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.models import Belief, Evidence, SourceKind


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
