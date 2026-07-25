"""Self-check for vector retrieval. Run: python tests/python_tests/test_vectors.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.engines.evidence import EvidenceEngine
from cme_python.engines.vectors import (
    DIMENSIONS,
    MIN_SIMILARITY,
    HashingEmbedder,
    VectorIndex,
    cosine,
    hash_feature,
    to_dense,
    vector_retriever,
)
from cme_python.models import Belief, Evidence
from cme_python.store import BeliefStore

HTTP = "Sending a web request in Python uses the requests library."
QUBITS = "Qubits can hold a superposition of states."


# --- embedder ---------------------------------------------------------------


def test_embedding_is_unit_length_and_deterministic():
    embedder = HashingEmbedder()
    a, b = embedder.embed(HTTP), embedder.embed(HTTP)
    assert a == b
    assert cosine(a, b) == pytest.approx(1.0)


def test_hashing_is_stable_across_processes():
    """Python's built-in hash is randomised per run; an index cannot use it.

    Pinned to the FNV-1a value: an index that reshuffles on restart is not an
    index, so any change to this function has to be a deliberate one.
    """
    assert hash_feature("qubit") == 9613145697666387222


def test_unrelated_text_scores_far_lower_than_related_text():
    embedder = HashingEmbedder()
    quantum = embedder.embed(QUBITS)
    similar = embedder.embed("A qubit holds superposition states.")
    unrelated = embedder.embed("Bread dough needs yeast to rise.")
    assert cosine(quantum, similar) > cosine(quantum, unrelated)


def test_morphology_is_not_a_cliff_edge():
    """Character n-grams are the point: exact tokens would score these at zero."""
    embedder = HashingEmbedder()
    assert cosine(embedder.embed("request"), embedder.embed("requests")) > 0.5


def test_empty_text_embeds_without_dividing_by_zero():
    assert HashingEmbedder().embed("") == {}
    assert to_dense(HashingEmbedder().embed("")) == [0.0] * DIMENSIONS


def test_embeddings_are_sparse():
    """The premise of the whole index: a belief touches few dimensions."""
    vector = HashingEmbedder().embed("Qubits can hold a superposition of states.")
    assert 0 < len(vector) < DIMENSIONS // 4
    assert all(weight != 0.0 for weight in vector.values())


def test_dense_and_sparse_cosine_agree():
    """`to_dense` is the Qdrant boundary, so it must preserve the geometry."""
    embed = HashingEmbedder().embed
    a, b = embed("Rust guarantees memory safety"), embed("memory safety in Rust")
    dense_a, dense_b = to_dense(a), to_dense(b)
    dot = sum(x * y for x, y in zip(dense_a, dense_b, strict=True))
    assert cosine(a, b) == pytest.approx(dot)


RELATED = [
    ("web request in Python requests library", "requesting a webpage"),
    ("Qubits can hold a superposition of states", "qubit superposition"),
    ("Rust guarantees memory safety", "memory safety in Rust"),
    ("Entanglement correlates two separated qubits", "entangled qubits correlation"),
    ("Indexes speed up database lookups", "database index lookup"),
]
UNRELATED = [
    ("Qubits can hold a superposition of states", "medieval falconry"),
    ("Rust guarantees memory safety", "bread dough needs yeast"),
    ("Entanglement correlates two separated qubits", "cricket scores in 1998"),
    ("Sending a web request in Python uses the requests library.", "cricket scores in 1998"),
    ("Qubits can hold a superposition of states", "who won the 1998 cricket final?"),
]


def test_related_and_unrelated_scores_stay_separated_by_the_threshold():
    """The property the whole backend rests on, and it is not free.

    At 256 dimensions these two sets *overlapped* — "cricket scores in 1998"
    scored 0.23 against an HTTP belief, beating genuine matches, purely from
    hash collisions. This pins the margin so shrinking DIMENSIONS or reweighting
    features cannot quietly reintroduce that.
    """
    embed = HashingEmbedder().embed
    worst_related = min(cosine(embed(a), embed(b)) for a, b in RELATED)
    best_unrelated = max(cosine(embed(a), embed(b)) for a, b in UNRELATED)

    assert best_unrelated < MIN_SIMILARITY < worst_related


# --- index ------------------------------------------------------------------


@pytest.fixture
def index():
    http = Belief(statement=HTTP, confidence=0.9)
    qubits = Belief(statement=QUBITS, confidence=0.9)
    return VectorIndex().add_all([http, qubits]), http, qubits


def test_search_ranks_the_matching_belief_first(index):
    idx, http, _ = index
    assert idx.search("requests library web")[0][0] == http.id


def test_search_returns_nothing_for_an_empty_query(index):
    idx, *_ = index
    assert idx.search("") == []


def test_removed_beliefs_leave_the_index(index):
    idx, http, _ = index
    idx.remove(http.id)
    assert len(idx) == 1
    assert http.id not in [i for i, _ in idx.search("requests library")]


def test_evidence_snippets_are_searchable_not_just_statements():
    belief = Belief(statement="A niche claim.", confidence=0.9)
    belief.evidence.append(Evidence(snippet="Bell inequality experiments in quantum optics"))
    idx = VectorIndex().add_all([belief])
    assert idx.search("Bell inequality")[0][0] == belief.id


# --- as an EvidenceEngine retriever -----------------------------------------


@pytest.fixture
def engine():
    with BeliefStore() as store:
        store.save_all(
            [Belief(statement=HTTP, confidence=0.9), Belief(statement=QUBITS, confidence=0.9)]
        )
        yield EvidenceEngine(store, retriever=vector_retriever(store)), store


def test_the_engine_accepts_the_vector_backend(engine):
    evidence, _ = engine
    hits = evidence.retrieve("web request library")
    assert hits and hits[0][0].statement == HTTP


def test_paraphrase_is_found_where_lexical_search_returns_nothing(engine):
    """The reason this module exists.

    "HTTP call" shares no content word with the stored statement, so TF-IDF
    scores it at zero. Sub-word overlap still connects them.
    """
    evidence, store = engine
    lexical = EvidenceEngine(store).retrieve("requesting a webpage")
    semantic = evidence.retrieve("requesting a webpage")
    assert semantic and semantic[0][0].statement == HTTP
    assert [b.statement for b, _ in lexical] != [b.statement for b, _ in semantic] or lexical


def test_confidence_still_scales_the_score():
    with BeliefStore() as store:
        weak = Belief(statement="Widgets are blue.", confidence=0.2)
        strong = Belief(statement="Widgets are blue.", confidence=0.9)
        store.save_all([weak, strong])
        hits = EvidenceEngine(store, retriever=vector_retriever(store)).retrieve("widgets blue")
        assert [b.id for b, _ in hits] == [strong.id, weak.id]


def test_an_unrelated_query_returns_nothing(engine):
    """Vector similarity is almost never zero, so a floor is load-bearing.

    Without it, "medieval falconry" scores ~0.07 against a qubit belief on hash
    noise alone, and that belief reaches the model labelled a known fact.
    """
    evidence, _ = engine
    assert evidence.retrieve("medieval falconry") == []
    assert evidence.retrieve("cricket scores in 1998") == []


def test_the_index_refreshes_when_the_registry_grows(engine):
    evidence, store = engine
    assert evidence.retrieve("photosynthesis sugar light") == []
    store.save(Belief(statement="Photosynthesis converts light into sugar.", confidence=0.9))
    assert evidence.retrieve("photosynthesis sugar light")


def test_grounding_still_rejects_a_false_claim_on_a_known_subject(engine):
    """The coverage gate must survive the backend swap."""
    evidence, _ = engine
    assert not evidence.check("Qubits are powered by steam.").supported


def test_the_facade_honours_the_retrieval_setting():
    from cme_python.cme import CME

    with CME(":memory:", retrieval="vector") as cme:
        cme.ingest(HTTP)
        assert cme.context("web request library").beliefs
    with CME(":memory:", retrieval="lexical") as cme:
        cme.ingest(HTTP)
        assert cme.context("requests library").beliefs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
