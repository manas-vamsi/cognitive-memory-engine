"""Self-check for the facade and the HTTP API.

Run: python tests/python_tests/test_api.py
"""

import sys
from dataclasses import replace
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
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_is_disabled_without_a_configured_model(client):
    """The rest of the service must work with no LLM, and say so clearly."""
    assert client.get("/health").json()["ask_enabled"] is False
    refused = client.post("/ask", json={"question": "anything?"})
    assert refused.status_code == 503
    assert "CME_LLM" in refused.json()["detail"]


def test_ask_returns_the_answer_with_its_context_and_verdict(monkeypatch):
    """With a connector wired in, /ask is the whole loop in one call."""
    from cme_python.clients.base import EchoClient, GroundedClient

    original = main.CME
    monkeypatch.setattr(main, "CME", lambda *a, **k: original(":memory:"))
    monkeypatch.setattr(
        main,
        "_build_chat",
        lambda cme: GroundedClient(cme, EchoClient(reply="Qubits are powered by steam.")),
    )
    with TestClient(main.app) as c:
        c.post("/ingest", json={"text": DOC, "locator": "doi:10/x"})
        assert c.get("/health").json()["ask_enabled"] is True

        body = c.post("/ask", json={"question": "how do qubits work?"}).json()
        assert body["answer"] == "Qubits are powered by steam."
        assert body["context"]["beliefs"]  # memories were attached
        assert body["report"]["checks"][0]["supported"] is False  # and it was caught


def test_the_timeline_endpoint_shows_how_a_belief_got_here(client):
    text = "Elixir runs on the BEAM virtual machine."
    belief_id = client.post("/ingest", json={"text": text, "locator": "doc:1"}).json()[0]["id"]
    client.post("/ingest", json={"text": text, "locator": "doc:2"})
    causes = [r["cause"] for r in client.get(f"/beliefs/{belief_id}/timeline").json()]
    # Re-ingesting reinforces rather than duplicates, and the timeline says so.
    assert causes == ["created", "merged"]
    assert client.get("/beliefs/ghost/timeline").status_code == 404


def test_superseding_takes_a_belief_out_of_recall(client):
    old = client.post("/ingest", json={"text": "The rate is 4 percent."}).json()[0]
    new = client.post("/ingest", json={"text": "The rate is 5 percent."}).json()[0]
    retired = client.post(f"/beliefs/{old['id']}/supersede", params={"replaced_by": new["id"]})
    assert retired.json()["superseded_by"] == new["id"]

    found = [
        b["id"] for b in client.post("/context", json={"query": "rate percent"}).json()["beliefs"]
    ]
    assert new["id"] in found
    assert old["id"] not in found
    # Retired, not deleted: the timeline is why the replacement is trusted.
    assert client.get(f"/beliefs/{old['id']}/timeline").json()[-1]["cause"] == "superseded"


def test_reconcile_leaves_a_dead_heat_for_a_human(client):
    """One document each is the common case, and it favours neither side."""
    client.post("/ingest", json={"text": "Rust has a garbage collector."})
    client.post("/ingest", json={"text": "Rust has no garbage collector."})
    assert len(client.get("/contradictions").json()) == 1

    assert client.post("/contradictions/reconcile").json() == []
    assert len(client.get("/contradictions").json()) == 1  # still flagged, untouched


def test_reconcile_endpoint_acts_on_what_contradictions_only_reports(client):
    client.post("/ingest", json={"text": "Rust has a garbage collector."})
    for locator in ("rust-lang.org", "the Rust Book", "the ownership RFC"):
        client.post("/ingest", json={"text": "Rust has no garbage collector.", "locator": locator})
    assert len(client.get("/contradictions").json()) == 1

    done = client.post("/contradictions/reconcile").json()
    assert len(done) == 1
    assert done[0]["retired"] is True
    assert client.get("/contradictions").json() == []  # the registry is consistent now
    # The retired claim is out of recall, and says why in its own timeline.
    assert client.get(f"/beliefs/{done[0]['loser']}/timeline").json()[-1]["cause"] == "superseded"


def test_health_and_memory_agree_on_the_size_of_the_registry(client):
    client.post("/ingest", json={"text": "Rust has a garbage collector."})
    for locator in ("rust-lang.org", "the Rust Book", "the ownership RFC"):
        client.post("/ingest", json={"text": "Rust has no garbage collector.", "locator": locator})
    client.post("/maintain")

    health, memory = client.get("/health").json(), client.get("/memory").json()
    assert health["beliefs"] == memory["total"]
    assert health["retired"] == memory["retired"] == 1
    assert memory["total"] == len(
        client.post("/context", json={"query": "garbage"}).json()["beliefs"]
    )


def test_maintain_does_the_three_upkeep_jobs_in_one_pass(cme):
    from datetime import UTC, datetime, timedelta

    from cme_python.models import Belief

    stale = Belief(statement="Nobody has mentioned this in a year.", confidence=0.8)
    stale.confidence_at = datetime.now(UTC) - timedelta(days=365)
    disproven = Belief(statement="Already disproven.", confidence=0.0)
    cme.memory.remember([stale, disproven])
    cme.ingest("Rust has a garbage collector.")
    for locator in ("rust-lang.org", "the Rust Book", "the ownership RFC"):
        cme.ingest("Rust has no garbage collector.", locator=locator)

    done = cme.maintain()
    assert done.decayed >= 1  # the stale belief aged
    assert done.retired == 1  # the outgunned claim was superseded
    assert done.pruned == 1  # the disproven one was cleared out
    assert done.changed == done.decayed + done.weakened + done.retired + done.pruned

    assert cme.maintain().changed == 0  # a second pass has nothing left to do


def test_maintain_never_deletes_a_belief_for_being_old(cme):
    """Decay stops well above the pruning threshold. Silence is not refutation."""
    from datetime import UTC, datetime, timedelta

    from cme_python.models import Belief

    ancient = Belief(statement="True but unfashionable for a decade.", confidence=0.8)
    ancient.confidence_at = datetime.now(UTC) - timedelta(days=3650)
    cme.memory.remember([ancient])

    assert cme.maintain().pruned == 0
    assert cme.store.get(ancient.id) is not None


def test_maintain_endpoint_reports_what_it_changed(client):
    client.post("/ingest", json={"text": "Rust has a garbage collector."})
    for locator in ("rust-lang.org", "the Rust Book", "the ownership RFC"):
        client.post("/ingest", json={"text": "Rust has no garbage collector.", "locator": locator})

    body = client.post("/maintain").json()
    assert body["retired"] == 1
    assert client.get("/contradictions").json() == []


def test_a_broken_connector_disables_ask_but_boots_the_server(monkeypatch):
    """An optional feature failing must not take the whole service down."""

    def explode(_name, _model=""):
        raise RuntimeError("The Claude client needs `anthropic`. Run `pip install anthropic`.")

    original = main.CME
    monkeypatch.setattr(main, "CME", lambda *a, **k: original(":memory:"))
    # Settings is frozen, so swap the whole object rather than a field.
    monkeypatch.setattr(main, "settings", replace(main.settings, llm="claude"))
    monkeypatch.setattr(main, "build_client", explode)

    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/health").json()["ask_enabled"] is False
        assert c.post("/ingest", json={"text": DOC}).status_code == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
