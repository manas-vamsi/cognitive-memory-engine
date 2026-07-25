"""Self-check for the facade and the HTTP API.

Run: python tests/python_tests/test_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient

from cme_python import main
from cme_python.cme import CME
from cme_python.models import SourceKind

DOC = (
    "Qubits can hold a superposition of states. "
    "Entanglement correlates two separated qubits. "
    "Rust guarantees memory safety without a garbage collector."
)


@pytest.fixture
def cme():
    with CME(":memory:") as engine:
        engine.ingest(DOC, source=SourceKind.RESEARCH_PAPER, locator="doi:10/x")
        yield engine


@pytest.fixture
def client(monkeypatch):
    """Run the app against an in-memory registry instead of a file."""
    original = main.CME
    monkeypatch.setattr(main, "CME", lambda *a, **k: original(":memory:"))
    with TestClient(main.app) as c:
        yield c


# --- facade ----------------------------------------------------------------


def test_ingest_files_every_claim(cme):
    assert len(cme.store) == 3


def test_context_returns_relevant_beliefs_with_their_proof(cme):
    ctx = cme.context("qubits superposition")
    assert ctx.beliefs
    assert len(ctx.justifications) == len(ctx.beliefs)
    assert all("superposition" in b.statement or "qubit" in b.statement for b in ctx.beliefs)
    assert "doi:10/x" in ctx.as_prompt()


def test_context_stays_inside_the_budget(cme):
    ctx = cme.context("qubits superposition entanglement", budget=6)
    assert 0 < ctx.tokens <= 6


def test_context_for_an_unknown_topic_is_empty(cme):
    ctx = cme.context("medieval falconry")
    assert ctx.beliefs == []
    assert ctx.as_prompt() == ""


def test_verify_flags_what_the_registry_cannot_back(cme):
    report = cme.verify(
        "Entanglement correlates two separated qubits. Qubits are powered by steam."
    )
    assert report.score == 0.5
    assert [c.claim for c in report.unsupported] == ["Qubits are powered by steam."]


def test_explain_returns_none_for_an_unknown_belief(cme):
    assert cme.explain("ghost") is None


def test_explain_answers_why_where_and_how_certain(cme):
    belief = cme.context("entanglement").beliefs[0]
    assert "doi:10/x" in cme.explain(belief.id).explain()


# --- http ------------------------------------------------------------------


def test_health_reports_the_registry_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["beliefs"] == 0


def test_ingest_context_and_verify_round_trip(client):
    filed = client.post(
        "/ingest",
        json={"text": DOC, "source": "research_paper", "locator": "doi:10/x"},
    )
    assert filed.status_code == 200
    assert len(filed.json()) == 3

    ctx = client.post("/context", json={"query": "qubits superposition"}).json()
    assert ctx["beliefs"]
    assert ctx["tokens"] > 0

    report = client.post("/verify", json={"answer": "Qubits are powered by steam."}).json()
    assert report["checks"][0]["supported"] is False


def test_ingesting_the_same_document_twice_does_not_duplicate(client):
    payload = {"text": DOC, "source": "book"}
    client.post("/ingest", json=payload)
    client.post("/ingest", json=payload)
    assert client.get("/health").json()["beliefs"] == 3


def test_explain_endpoint_and_its_404(client):
    client.post("/ingest", json={"text": DOC, "locator": "doi:10/x"})
    belief_id = client.post("/context", json={"query": "entanglement"}).json()["beliefs"][0]["id"]

    body = client.get(f"/beliefs/{belief_id}").json()
    assert body["supporting"][0]["locator"] == "doi:10/x"

    assert client.get("/beliefs/ghost").status_code == 404


def test_contradictions_endpoint_reports_a_clash(client):
    client.post("/ingest", json={"text": "Rust has a garbage collector."})
    client.post("/ingest", json={"text": "Rust has no garbage collector."})
    found = client.get("/contradictions").json()
    assert len(found) == 1
    assert "contradicts" in found[0]["explanation"]
    assert found[0]["winner"] != found[0]["loser"]


def test_empty_payloads_are_rejected(client):
    assert client.post("/ingest", json={"text": ""}).status_code == 422
    assert client.post("/context", json={"query": ""}).status_code == 422
    assert client.post("/context", json={"query": "x", "budget": 0}).status_code == 422
    assert client.post("/verify", json={"answer": ""}).status_code == 422


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
