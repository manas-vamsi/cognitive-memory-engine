"""Self-check for contradiction detectors. Run: python tests/python_tests/test_entailment.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.entailment import NLIDetector, get_detector
from cme_python.engines.reasoning import LexicalDetector, ReasoningEngine
from cme_python.models import Belief
from cme_python.store import BeliefStore

HAS_GC = "Rust has a garbage collector."
NO_GC = "Rust has no garbage collector."

# The pair the lexical detector exists to catch is easy. This is the pair it
# cannot: a genuine disagreement with no negation word in either sentence.
SAFE = "Rust is memory-safe."
LEAKS = "Rust leaks memory constantly."


def _has(module: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is not None


needs_model = pytest.mark.skipif(
    not (_has("transformers") and _has("torch")),
    reason="the NLI detector needs transformers and torch",
)


def test_the_default_detector_is_the_one_that_needs_nothing():
    assert isinstance(get_detector(), LexicalDetector)
    assert isinstance(get_detector("lexical"), LexicalDetector)


def test_an_unknown_detector_says_what_there_is():
    with pytest.raises(ValueError, match="lexical"):
        get_detector("telepathy")


def test_asking_for_the_nli_detector_downloads_nothing_until_it_is_used():
    """Constructing an engine must not pull a model off the internet."""
    detector = get_detector("nli")
    assert isinstance(detector, NLIDetector)
    assert detector._pipe is None


def test_the_engine_takes_whatever_detector_it_is_given():
    """The seam itself, proved without a model in sight."""

    class AlwaysClashes:
        def clashes(self, beliefs, threshold):
            return [(beliefs[0], beliefs[1], 0.99)]

    with BeliefStore() as store:
        store.save_all([Belief(statement="The sky is blue."), Belief(statement="Water is wet.")])
        found = ReasoningEngine(store, detector=AlwaysClashes()).contradictions()

        assert len(found) == 1
        assert found[0].overlap == 0.99
        # And the rest of the engine acts on it, detector regardless.
        assert found[0].winner is not found[0].loser


def test_the_lexical_detector_cannot_see_a_reworded_disagreement():
    """The gap this module exists to close, stated as a test rather than a comment."""
    with BeliefStore() as store:
        store.save_all([Belief(statement=SAFE), Belief(statement=LEAKS)])
        assert ReasoningEngine(store, detector=LexicalDetector()).contradictions() == []

        # The same engine still catches the negated form it was built for.
        store.save_all([Belief(statement=HAS_GC), Belief(statement=NO_GC)])
        assert len(ReasoningEngine(store, detector=LexicalDetector()).contradictions()) == 1


@needs_model
def test_the_nli_detector_sees_what_negation_parity_misses():
    """The whole point: a contradiction with no negation word in either sentence."""
    with BeliefStore() as store:
        store.save_all([Belief(statement=SAFE), Belief(statement=LEAKS)])
        found = ReasoningEngine(store, detector=get_detector("nli")).contradictions(threshold=0.5)

        assert len(found) == 1
        assert {found[0].a.statement, found[0].b.statement} == {SAFE, LEAKS}


@needs_model
def test_the_nli_detector_leaves_agreeing_beliefs_alone():
    """A detector that flags everything would pass the test above and be useless."""
    with BeliefStore() as store:
        store.save_all(
            [
                Belief(statement=SAFE),
                Belief(statement="Rust prevents data races at compile time."),
            ]
        )
        assert (
            ReasoningEngine(store, detector=get_detector("nli")).contradictions(threshold=0.5) == []
        )


@needs_model
def test_the_nli_detector_is_never_asked_about_unrelated_beliefs():
    """The failure this gate exists for, pinned so it cannot come back.

    Asked directly, the model calls "Rust is memory-safe" and "Qubits can hold a
    superposition" a contradiction at 0.97 — unrelated pairs are barely in an
    NLI model's training distribution and it reaches for contradiction rather
    than neutral. Ungated, the detector reports most of a registry as
    self-contradictory, confidently.
    """
    unrelated = "Qubits can hold a superposition of states."
    with BeliefStore() as store:
        store.save_all([Belief(statement=SAFE), Belief(statement=unrelated)])
        assert (
            ReasoningEngine(store, detector=get_detector("nli")).contradictions(threshold=0.5) == []
        )


REGISTRY = [
    "Rust guarantees memory safety without a garbage collector.",
    "The borrow checker rejects programs that alias mutable state.",
    "Rust was first released in 2015 by Mozilla Research.",
    "Cargo is the package manager for Rust.",
    "Python is commonly used in machine learning.",
    "The global interpreter lock prevents two threads from running bytecode at once.",
    "Qubits can hold a superposition of states.",
    "Decoherence limits how long a quantum state survives.",
    "PostgreSQL uses multiversion concurrency control.",
    "Vacuum reclaims space held by rows that are no longer visible.",
    "Replication streams the log to standby servers.",
    "Docker packages an application with its dependencies.",
    "TLS encrypts traffic between a client and a server.",
]

DISAGREEMENTS = {
    "The borrow checker permits programs that alias mutable state.": 1,
    "The global interpreter lock lets two threads run bytecode at once.": 5,
    "Decoherence extends how long a quantum state survives.": 7,
    "Vacuum reclaims space held by rows that are still visible.": 9,
}


@needs_model
def test_the_detector_is_precise_over_a_whole_registry():
    """Two or three beliefs cannot show what this does to a real one.

    `reconcile` retires beliefs off the back of these pairs, so a false
    positive is not a noisy report — it takes a true belief out of recall. At a
    looser gate this registry produced two: "TLS encrypts traffic between a
    client and a server" against "Replication streams the log to standby
    servers", on the strength of the word *server*.
    """
    with BeliefStore() as store:
        beliefs = [Belief(statement=s) for s in REGISTRY]
        expected = set()
        for text, target in DISAGREEMENTS.items():
            beliefs.append(Belief(statement=text))
            expected.add(frozenset((beliefs[-1].id, beliefs[target].id)))
        store.save_all(beliefs)

        found = ReasoningEngine(store, detector=get_detector("nli")).contradictions(threshold=0.5)
        pairs = {frozenset((c.a.id, c.b.id)) for c in found}

        assert pairs == expected  # every planted one, and nothing else


def test_the_lexical_detector_misses_these_and_that_is_the_point():
    """The same registry through the default detector, for the contrast.

    None of the disagreements above uses a negation word, so parity cannot see
    them. This is what the model is for, and what it costs a model to buy.
    """
    with BeliefStore() as store:
        store.save_all([Belief(statement=s) for s in REGISTRY + list(DISAGREEMENTS)])
        found = ReasoningEngine(store, detector=LexicalDetector()).contradictions()
        assert len(found) < len(DISAGREEMENTS)


@needs_model
def test_the_gate_is_a_precondition_not_a_detector():
    """It must not be doing the work — pairs it admits still have to be judged."""
    detector = get_detector("nli")
    agreeing = [Belief(statement=SAFE), Belief(statement="Rust is a safe language.")]

    # Same subject matter, so the gate lets them through to the model...
    from cme_python.engines.entailment import _overlap
    from cme_python.engines.reasoning import _content

    assert _overlap(_content(agreeing[0].statement), _content(agreeing[1].statement)) >= 0.1
    # ...and the model is what decides they do not clash.
    assert detector.clashes(agreeing, 0.5) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
