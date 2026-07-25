"""Self-check for the Evidence Engine. Run: python tests/python_tests/test_evidence.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.evidence import EvidenceEngine, tokenise
from cme_python.models import Belief, Evidence, SourceKind
from cme_python.store import BeliefStore

QUBITS = "Qubits can hold a superposition of states."
RUST = "Rust guarantees memory safety without a garbage collector."


@pytest.fixture
def engine():
    with BeliefStore() as store:
        q = Belief(statement=QUBITS, confidence=0.9)
        q.evidence.append(
            Evidence(
                snippet="Bell inequality experiments",
                source=SourceKind.RESEARCH_PAPER,
                locator="doi:10/x",
            )
        )
        r = Belief(statement=RUST, confidence=0.95)
        r.evidence.append(
            Evidence(snippet="The Rust Book, ownership chapter", locator="doc.rust-lang.org")
        )
        store.save_all([q, r])
        yield EvidenceEngine(store)


def test_tokenise_drops_stopwords_and_single_chars():
    assert tokenise("A qubit is in the state of X") == ["qubit", "state"]


def test_retrieval_ranks_the_relevant_belief_first(engine):
    hits = engine.retrieve("garbage collector memory safety")
    assert hits[0][0].statement == RUST
    assert hits[0][1] > 0


def test_retrieval_matches_on_evidence_snippets_not_just_statements(engine):
    hits = engine.retrieve("Bell inequality")
    assert hits and hits[0][0].statement == QUBITS


def test_unrelated_and_empty_queries_return_nothing(engine):
    assert engine.retrieve("cricket scores in 1998") == []
    assert engine.retrieve("the of and") == []


def test_confidence_breaks_ties_between_equally_relevant_beliefs():
    with BeliefStore() as store:
        weak = Belief(statement="Widgets are blue.", confidence=0.2)
        strong = Belief(statement="Widgets are blue.", confidence=0.9)
        store.save_all([weak, strong])
        hits = EvidenceEngine(store).retrieve("widgets blue")
        assert [b.id for b, _ in hits] == [strong.id, weak.id]


def test_index_refreshes_when_the_registry_grows(engine):
    assert engine.retrieve("photosynthesis") == []
    engine.store.save(Belief(statement="Photosynthesis converts light into sugar.", confidence=0.9))
    assert engine.retrieve("photosynthesis")


def test_an_edited_belief_is_reindexed(engine):
    """The registry size does not change on an edit, so the old check missed it.

    Comparing `len(store)` meant a belief could be rewritten and retrieval would
    keep answering from the previous text indefinitely.
    """
    belief = engine.retrieve(RUST)[0][0]
    belief.statement = "Rust now discusses photosynthesis and chlorophyll instead."
    engine.store.save(belief)

    assert engine.retrieve("photosynthesis chlorophyll")
    assert engine.retrieve("garbage collector") == []


def test_an_equal_sized_add_and_delete_is_noticed(engine):
    """Count stays the same, contents do not — also missed by a length check."""
    doomed = engine.retrieve(RUST)[0][0]
    engine.store.delete(doomed.id)
    engine.store.save(Belief(statement="Photosynthesis converts light into sugar.", confidence=0.9))

    assert len(engine.store) == 2
    assert engine.retrieve("photosynthesis sugar")
    assert engine.retrieve("garbage collector") == []


def test_a_deleted_belief_leaves_the_index(engine):
    doomed = engine.retrieve(RUST)[0][0]
    engine.store.delete(doomed.id)
    assert engine.retrieve("garbage collector") == []


def test_the_incremental_index_matches_a_full_rebuild(engine):
    """Cheap must also mean identical, or the optimisation is a bug."""
    engine.store.save(
        Belief(statement="Entanglement correlates two separated qubits.", confidence=0.9)
    )
    engine.store.save(Belief(statement="Indexes speed up database lookups.", confidence=0.7))
    incremental = engine.retrieve("qubits entanglement superposition", limit=5)

    rebuilt = EvidenceEngine(engine.store)
    rebuilt.reindex()
    full = rebuilt.retrieve("qubits entanglement superposition", limit=5)

    assert [b.id for b, _ in incremental] == [b.id for b, _ in full]
    for (_, a), (_, b) in zip(incremental, full, strict=True):
        assert a == pytest.approx(b)


def test_postings_shortcut_matches_a_full_scan(engine):
    """Skipping non-matching beliefs must change speed, not answers.

    Scores every belief the slow way and asserts the indexed path agrees —
    ordering, membership and scores.
    """
    for i in range(25):
        engine.store.save(
            Belief(
                statement=f"Belief {i} about qubits, rust, or databases.", confidence=0.5 + i / 100
            )
        )
    engine._fresh_index()

    query = "qubits rust databases"
    terms = tokenise(query)
    brute = []
    for belief_id, doc in engine._docs.items():
        overlap = sum(doc[t] * engine._idf.get(t, 0.0) for t in terms)
        if overlap <= 0:
            continue
        belief = engine.store.get(belief_id)
        brute.append(
            (belief, round((overlap / (sum(doc.values()) or 1) ** 0.5) * belief.confidence, 6))
        )
    brute.sort(key=lambda pair: pair[1], reverse=True)

    indexed = engine.retrieve(query, limit=len(brute))
    assert [b.id for b, _ in indexed] == [b.id for b, _ in brute[: len(indexed)]]
    for (_, a), (_, b) in zip(indexed, brute, strict=False):
        assert a == pytest.approx(b)


def test_postings_drop_terms_when_a_belief_is_edited_away(engine):
    """A term nobody mentions any more must stop matching."""
    belief = engine.retrieve(RUST)[0][0]
    belief.statement = "Rust now discusses photosynthesis instead."
    engine.store.save(belief)

    assert engine.retrieve("garbage collector") == []
    assert "collector" not in engine._postings


def test_justify_separates_support_from_contradiction(engine):
    b = engine.retrieve(RUST)[0][0]
    assert engine.justify(b).verdict == "grounded"

    b.add_evidence(Evidence(snippet="unsafe blocks exist", strength=0.5, supports=False))
    j = engine.justify(b)
    assert j.verdict == "disputed"
    assert len(j.supporting) == 1 and len(j.contradicting) == 1
    assert "doc.rust-lang.org" in j.explain()


def test_justify_reports_a_belief_with_no_evidence_as_unsupported(engine):
    assert engine.justify(Belief(statement="Bare assertion.")).verdict == "unsupported"


def test_grounding_flags_the_invented_sentence(engine):
    report = engine.ground(
        "Rust guarantees memory safety without a garbage collector. "
        "The moon is made of green cheese and orbits Jupiter."
    )
    assert len(report.checks) == 2
    assert report.checks[0].supported
    assert report.checks[0].belief is not None
    assert not report.checks[1].supported
    assert report.checks[1].belief is None
    assert report.is_grounded is False
    assert report.score == 0.5


def test_fully_grounded_text_passes(engine):
    report = engine.ground(RUST)
    assert report.is_grounded and report.score == 1.0


def test_short_claims_are_verified_not_skipped(engine):
    """A fragment too short to become a belief is still checked for grounding."""
    report = engine.ground("Rust is safe. Cheese orbits Jupiter.")
    assert [c.claim for c in report.checks] == ["Rust is safe.", "Cheese orbits Jupiter."]
    assert [c.supported for c in report.checks] == [True, False]


def test_a_false_claim_on_a_known_subject_is_not_grounded(engine):
    """Sharing a subject word is not evidence — this is the guard that matters."""
    check = engine.check("Qubits are powered by steam.")
    assert check.relevance > 0  # topically it looks relevant
    assert check.coverage < 0.5  # but the belief accounts for almost none of it
    assert not check.supported


def test_questions_are_not_treated_as_claims(engine):
    report = engine.ground("Is Rust memory safe? " + RUST)
    assert [c.claim for c in report.checks] == [RUST]


def test_a_custom_retriever_replaces_the_lexical_default():
    with BeliefStore() as store:
        b = store.save(Belief(statement="Anything.", confidence=0.5))
        engine = EvidenceEngine(store, retriever=lambda q, limit: [(b, 0.99)])
        assert engine.retrieve("ignored")[0][1] == 0.99
        assert engine.check("Anything.").supported


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
