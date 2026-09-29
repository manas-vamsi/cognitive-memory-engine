"""Self-check for multi-hop context: concepts in, graph walk, conflicts out.

Run: python tests/python_tests/test_multihop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.cme import CME
from cme_python.engines.belief import concepts_in
from cme_python.models import MemoryTier

PEOPLE = "Alice Moreau was born in Lyon. Bruno Keller was born in Tarsk."
CITIES = "Lyon is a city in Veloria. Tarsk is a city in Dornland."
EXPORTS = "Veloria mainly exports copper. Dornland mainly exports timber."
QUESTION = "What does the country where Alice Moreau was born export?"
CHAIN = {
    "Alice Moreau was born in Lyon.",
    "Lyon is a city in Veloria.",
    "Veloria mainly exports copper.",
}


@pytest.fixture
def cme():
    with CME(":memory:") as engine:
        for doc in (PEOPLE, CITIES, EXPORTS):
            engine.ingest(doc)
        yield engine


def statements(context) -> set[str]:
    return {b.statement for b in context.beliefs}


def test_concepts_are_the_named_things_in_a_claim():
    assert concepts_in("Alice Moreau was born in Lyon.") == ["Alice Moreau", "Lyon"]


def test_function_words_never_become_concepts():
    assert concepts_in("The Rhine flows through Basel.") == ["Rhine", "Basel"]
    assert concepts_in("It is raining.") == []


def test_each_concept_is_reported_once():
    assert concepts_in("Lyon is near Lyon.") == ["Lyon"]


def test_ingest_files_concepts_as_connections(cme):
    (belief,) = [b for b in cme.store.all() if b.statement.startswith("Lyon")]
    assert belief.connections == {"Lyon", "Veloria"}


def test_retrieval_alone_cannot_answer_a_two_hop_question(cme):
    """The baseline this whole feature exists to beat."""
    assert not statements(cme.context(QUESTION, hops=0)) >= CHAIN


def test_walking_the_graph_finds_every_link_in_the_chain(cme):
    assert statements(cme.context(QUESTION, hops=2)) >= CHAIN


def test_one_hop_is_what_a_two_fact_question_takes(cme):
    question = "Which country was Alice Moreau born in?"
    assert "Lyon is a city in Veloria." not in statements(cme.context(question, hops=0))
    assert "Lyon is a city in Veloria." in statements(cme.context(question, hops=1))


def test_a_rare_concept_carries_more_than_a_common_one():
    """Kelp links one other belief, Moss eight: Moss must count for less."""
    with CME(":memory:") as engine:
        engine.ingest("Nora Vale studies Kelp and Moss.")
        engine.ingest("Kelp grows along the coast of Brel.")
        for n in range(8):
            engine.ingest(f"Moss grows on the walls of Town{n}.")
        for n in range(10):  # so Moss is common, not universal
            engine.ingest(f"Ferry{n} crosses the strait at dawn.")
        seeds = engine.evidence.retrieve("Nora Vale", limit=1)
        expanded = engine.reasoning.expand(seeds, hops=1)
        carried = {c.belief.statement: c.relevance for c in expanded}
        assert (
            carried["Moss grows on the walls of Town0."]
            < carried["Kelp grows along the coast of Brel."]
        )


def test_a_concept_most_of_the_registry_shares_is_not_walked():
    """Every belief mentions Note: crossing it would reach everything, meaning nothing."""
    with CME(":memory:") as engine:
        for n in range(6):
            engine.ingest(f"Note {n} says the Harbor{n} gate opens at noon.")
        seeds = engine.evidence.retrieve("Harbor0 gate", limit=1)
        reached = {c.belief.statement for c in engine.reasoning.expand(seeds, hops=1)}
        assert reached == {"Note 0 says the Harbor0 gate opens at noon."}


def test_the_walk_stays_inside_the_requested_scope(cme):
    cme.ingest("Veloria is governed from Brel.", tier=MemoryTier.USER, scope="someone-else")
    found = statements(cme.context(QUESTION, hops=2, tier=MemoryTier.GENERAL))
    assert "Veloria is governed from Brel." not in found


def test_expanded_beliefs_come_back_fresh_from_the_registry(cme):
    """The graph holds a snapshot; a belief reinforced since must not be stale."""
    assert len(cme.reasoning.graph)  # build the snapshot
    cme.ingest(CITIES)  # reinforce, same count, so the graph is not rebuilt
    fresh = {b.id: b.confidence for b in cme.store.all()}
    for belief in cme.context(QUESTION, hops=2).beliefs:
        assert belief.confidence == fresh[belief.id]


def test_contradictions_in_the_chosen_set_are_flagged_in_the_prompt():
    with CME(":memory:") as engine:
        engine.ingest("Veloria exports copper to Dornland.")
        engine.ingest("Veloria never exports copper to Dornland.")
        context = engine.context("Does Veloria export copper to Dornland?")
        assert len(context.conflicts) == 1
        assert "contradict each other" in context.as_prompt()


def test_a_consistent_context_has_no_conflicts(cme):
    context = cme.context(QUESTION)
    assert context.conflicts == []
    assert "contradict" not in context.as_prompt()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
