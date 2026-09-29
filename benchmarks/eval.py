"""Evaluate the research questions that need an answer, not a timing.

  RQ1  Can a system reason over beliefs instead of raw text?
       Same token budget, same questions: retrieve raw passages (what a
       retrieval-augmented pipeline does) versus CME's selected beliefs.
  RQ5  Does graph reasoning improve multi-step accuracy?
       The same beliefs with and without the Knowledge Graph walk, on
       questions one, two and three facts deep.
  RQ3  Does evidence tracking reduce hallucination?
       Needs a model. With CME_LLM set, every question is asked with no memory
       and with CME's, and each answer is scored and checked by `verify`.

The corpus is generated: invented people, cities and countries, so no model
can answer from what it already knows and every correct answer has to come
from memory. That is the point, and also the limit: these are clean facts in
plain sentences, and real documents are messier.

Run: python benchmarks/eval.py [--people 60] [--budget 40] [--llm claude]
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cme_python.cme import CME
from cme_python.engines.evidence import tokenise

SYLLABLES = [
    "ka",
    "lo",
    "ve",
    "ri",
    "ta",
    "mon",
    "sel",
    "dor",
    "pe",
    "lia",
    "bru",
    "osk",
    "tar",
    "zen",
    "fi",
    "qua",
    "mar",
    "nel",
    "vo",
    "ish",
]
GOODS = [
    "copper",
    "timber",
    "wool",
    "salt",
    "amber",
    "tea",
    "glass",
    "silk",
    "iron",
    "rice",
    "cotton",
    "olives",
]


def invented(rng: random.Random, taken: set[str], parts: int) -> str:
    while True:
        name = "".join(rng.choice(SYLLABLES) for _ in range(parts)).capitalize()
        if name not in taken:
            taken.add(name)
            return name


@dataclass
class Question:
    text: str
    answer: str
    facts: list[str]
    """The sentences an answer needs, in the order a reader would use them."""

    @property
    def hops(self) -> int:
        return len(self.facts)


@dataclass
class Corpus:
    documents: list[str] = field(default_factory=list)
    """Paragraphs as a document would hold them: a few facts each, mixed."""
    questions: list[Question] = field(default_factory=list)


def build_corpus(people: int, seed: int = 0) -> Corpus:
    rng = random.Random(seed)
    taken: set[str] = set()
    countries = [invented(rng, taken, 3) for _ in range(max(people // 5, 3))]
    cities = [invented(rng, taken, 2) for _ in range(max(people // 2, 6))]
    exports = {country: GOODS[i % len(GOODS)] for i, country in enumerate(countries)}
    city_in = {city: rng.choice(countries) for city in cities}

    born, located, exported = [], [], []
    corpus = Corpus()
    for _ in range(people):
        person = f"{invented(rng, taken, 2)} {invented(rng, taken, 2)}"
        city = rng.choice(cities)
        country = city_in[city]
        fact_born = f"{person} was born in {city}."
        fact_city = f"{city} is a city in {country}."
        fact_export = f"{country} mainly exports {exports[country]}."
        born.append(fact_born)
        corpus.questions += [
            Question(f"Where was {person} born?", city, [fact_born]),
            Question(f"Which country was {person} born in?", country, [fact_born, fact_city]),
            Question(
                f"What does the country where {person} was born export?",
                exports[country],
                [fact_born, fact_city, fact_export],
            ),
        ]
    located = [f"{city} is a city in {country}." for city, country in city_in.items()]
    exported = [f"{country} mainly exports {good}." for country, good in exports.items()]

    # Three facts to a paragraph, shuffled, the way a registry or a gazetteer
    # actually reads: the passage holding a fact also holds unrelated ones.
    for facts in (born, located, exported):
        facts = list(facts)
        rng.shuffle(facts)
        corpus.documents += [" ".join(facts[i : i + 3]) for i in range(0, len(facts), 3)]
    rng.shuffle(corpus.questions)
    return corpus


# --- the systems under test ------------------------------------------------


def cost(text: str) -> int:
    """The same token proxy the Optimization Engine budgets with."""
    return max(len(tokenise(text)), 1)


class PassageRetrieval:
    """Top passages by TF-IDF until the budget is spent. The usual baseline.

    Scored by CME's own Evidence Engine over whole paragraphs, so the only
    difference from the belief systems is the unit of memory, not the ranker.
    """

    def __init__(self, documents: list[str]) -> None:
        from cme_python.engines.evidence import EvidenceEngine  # noqa: PLC0415
        from cme_python.models import Belief  # noqa: PLC0415
        from cme_python.store import BeliefStore  # noqa: PLC0415

        self.store = BeliefStore()
        self.store.save_all([Belief(statement=d) for d in documents])
        self.evidence = EvidenceEngine(self.store)

    def context(self, query: str, budget: float) -> list[str]:
        chosen, spent = [], 0
        for passage, _ in self.evidence.retrieve(query, limit=50):
            if spent + cost(passage.statement) > budget:
                continue
            chosen.append(passage.statement)
            spent += cost(passage.statement)
        return chosen


def belief_context(cme: CME, query: str, budget: float, hops: int) -> list[str]:
    return [b.statement for b in cme.context(query, budget=budget, hops=hops).beliefs]


# --- scoring ---------------------------------------------------------------


@dataclass
class Score:
    recall: float = 0.0
    """Share of the needed facts that made it into the context."""
    complete: int = 0
    """Questions whose context held every fact the answer needs."""
    tokens: int = 0
    n: int = 0

    def add(self, question: Question, context: list[str]) -> None:
        text = " ".join(context)
        present = sum(fact in text for fact in question.facts)
        self.recall += present / len(question.facts)
        self.complete += present == len(question.facts)
        self.tokens += sum(cost(c) for c in context)
        self.n += 1

    def row(self) -> str:
        n = self.n or 1
        return (
            f"{self.recall / n:>7.0%} {self.complete / n:>9.0%} {self.tokens / n:>7.1f}"
            f" {self.recall / max(self.tokens, 1) * 100:>13.2f}"
        )


def evaluate_context(corpus: Corpus, budget: float) -> None:
    passages = PassageRetrieval(corpus.documents)
    with CME(":memory:") as cme:
        for doc in corpus.documents:
            cme.ingest(doc)
        systems = {
            "passages": lambda q: passages.context(q, budget),
            "beliefs": lambda q: belief_context(cme, q, budget, hops=0),
            "beliefs+graph": lambda q: belief_context(cme, q, budget, hops=2),
        }
        scores = {(name, h): Score() for name in systems for h in (1, 2, 3)}
        for question in corpus.questions:
            for name, run in systems.items():
                scores[(name, question.hops)].add(question, run(question.text))

    print(f"\n## Context quality at a {budget:g}-token budget (RQ1, RQ5)\n")
    print("  Recall: share of the needed facts in context. Complete: every one of")
    print("  them there, which is what answering takes. Recall per 100 tokens is")
    print("  what the budget bought.\n")
    print(f"  {'facts needed':<13} {'system':<14} {'recall':>7} {'complete':>9} {'tokens':>7}")
    print(f"  {'':<13} {'':<14} {'':>7} {'':>9} {'':>7} {'recall/100tok':>13}")
    for hops in (1, 2, 3):
        for name in systems:
            label = str(hops) if name == "passages" else ""
            print(f"  {label:<13} {name:<14} {scores[(name, hops)].row()}")


# --- RQ3: with a model ------------------------------------------------------


def evaluate_answers(corpus: Corpus, budget: float, llm: str, limit: int) -> None:
    from cme_python.clients.base import SYSTEM, GroundedClient, build_client  # noqa: PLC0415

    print(f"\n## Answers from {llm}, {limit} questions (RQ3)\n")
    client = build_client(llm)
    with CME(":memory:") as cme:
        for doc in corpus.documents:
            cme.ingest(doc)
        grounded = GroundedClient(cme, client)
        rows = {"no memory": [0, 0, 0], "CME": [0, 0, 0]}
        for question in corpus.questions[:limit]:
            bare = client.complete(question.text, system=SYSTEM)
            with_memory = grounded.ask(question.text, budget=budget).answer
            for name, answer in (("no memory", bare), ("CME", with_memory)):
                report = cme.verify(answer)
                correct = question.answer.lower() in answer.lower()
                rows[name][0] += correct
                # Wrong and still asserting something: a claim nothing backs.
                rows[name][1] += (not correct) and bool(report.unsupported)
                rows[name][2] += not report.unsupported
    print(f"  {'':<10} {'correct':>8} {'confidently wrong':>18} {'fully grounded':>15}")
    for name, (correct, wrong, clean) in rows.items():
        print(f"  {name:<10} {correct / limit:>8.0%} {wrong / limit:>18.0%} {clean / limit:>15.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=60, help="people in the corpus")
    parser.add_argument("--budget", type=float, default=40, help="context tokens")
    parser.add_argument("--llm", default="", help="connector for RQ3, e.g. claude")
    parser.add_argument("--questions", type=int, default=30, help="questions for RQ3")
    args = parser.parse_args()

    corpus = build_corpus(args.people)
    print(
        f"CME eval: {args.people} people, {len(corpus.documents)} passages, "
        f"{len(corpus.questions)} questions"
    )
    evaluate_context(corpus, args.budget)
    llm = args.llm or __import__("cme_python.config").config.settings.llm
    if llm:
        evaluate_answers(corpus, args.budget, llm, args.questions)
    else:
        print("\n## Answers (RQ3)\n\n  Skipped: pass --llm claude (or set CME_LLM) to run.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
