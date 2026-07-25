"""Self-check for belief splitting. Run: python tests/python_tests/test_belief_split.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.belief import BeliefEngine, split_claims
from cme_python.models import Belief, Evidence, MemoryTier
from cme_python.store import BeliefStore

TWO_CLAIMS = "Rust is fast and has no garbage collector."


# --- detection --------------------------------------------------------------


def test_a_conjunction_of_two_claims_splits_and_carries_the_subject():
    assert split_claims(TWO_CLAIMS) == [
        "Rust is fast.",
        "Rust has no garbage collector.",
    ]


def test_a_semicolon_joins_two_independent_claims():
    assert split_claims("Rust is compiled; Python is interpreted.") == [
        "Rust is compiled.",
        "Python is interpreted.",
    ]


def test_a_single_claim_does_not_split():
    assert split_claims("Qubits can hold a superposition of states.") == []


def test_a_conjoined_noun_phrase_is_not_two_claims():
    """The failure worth preventing: splitting this invents a claim.

    "guaranteed" looks verb-like by suffix, and suffix-guessing would produce
    the nonsense "Rust guaranteed thread safety".
    """
    assert split_claims("Rust gives zero-cost abstractions and guaranteed thread safety.") == []


def test_a_fragment_after_the_conjunction_does_not_split():
    assert split_claims("Rust is fast and safe.") == []


def test_questions_and_empty_input_are_left_alone():
    assert split_claims("") == []
    assert split_claims("and") == []


# --- the model --------------------------------------------------------------


def test_each_part_inherits_evidence_confidence_and_tier():
    belief = Belief(statement=TWO_CLAIMS, confidence=0.8, tier=MemoryTier.PROJECT, scope="acme")
    belief.connect("Rust")
    belief.evidence.append(Evidence(snippet="The Rust Book", locator="doc.rust-lang.org"))

    parts = belief.split(*split_claims(TWO_CLAIMS))
    assert [p.statement for p in parts] == ["Rust is fast.", "Rust has no garbage collector."]
    for part in parts:
        assert part.confidence == 0.8
        assert part.connections == {"Rust"}
        assert part.tier == MemoryTier.PROJECT and part.scope == "acme"
        assert [e.locator for e in part.evidence] == ["doc.rust-lang.org"]
    assert parts[0].id != parts[1].id


def test_parts_carry_independent_evidence_after_the_split():
    """The whole point: each claim can now be refuted on its own."""
    belief = Belief(statement=TWO_CLAIMS, confidence=0.8)
    belief.evidence.append(Evidence(snippet="shared", strength=0.5))
    first, second = belief.split(*split_claims(TWO_CLAIMS))

    second.add_evidence(Evidence(snippet="actually it has one", strength=0.9, supports=False))
    assert second.confidence < first.confidence
    assert len(first.evidence) == 1  # untouched


def test_splitting_on_fewer_than_two_statements_is_a_no_op():
    belief = Belief(statement=TWO_CLAIMS)
    assert belief.split() == [belief]
    assert belief.split("only one") == [belief]


# --- the engine -------------------------------------------------------------


@pytest.fixture
def engine():
    with BeliefStore() as store:
        yield BeliefEngine(store)


def test_split_replaces_the_original_in_the_registry(engine):
    original = engine.store.save(Belief(statement=TWO_CLAIMS, confidence=0.8))
    parts = engine.split(original)

    assert len(parts) == 2
    assert engine.store.get(original.id) is None
    assert sorted(b.statement for b in engine.store.all()) == [
        "Rust has no garbage collector.",
        "Rust is fast.",
    ]


def test_a_belief_that_cannot_split_is_left_in_place(engine):
    """A no-op split must not delete the belief it declined to change."""
    original = engine.store.save(Belief(statement="Qubits can hold a superposition."))
    assert engine.split(original) == [original]
    assert engine.store.get(original.id) is not None
    assert len(engine.store) == 1


def test_split_all_sweeps_only_what_needs_splitting(engine):
    engine.store.save_all(
        [
            Belief(statement=TWO_CLAIMS),
            Belief(statement="Qubits can hold a superposition of states."),
        ]
    )
    engine.split_all()
    assert len(engine.store) == 3  # one split in two, one left alone


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
