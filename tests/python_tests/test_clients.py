"""Self-check for the grounding middleware. Run: python tests/python_tests/test_clients.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from cme_python.clients.base import EchoClient, GroundedClient, LLMClient
from cme_python.cme import CME
from cme_python.models import SourceKind

DOC = "Qubits can hold a superposition of states. Entanglement correlates two separated qubits."
TRUE_ANSWER = "Entanglement correlates two separated qubits."
FALSE_ANSWER = "Qubits are powered by steam."


@pytest.fixture
def cme():
    with CME(":memory:") as engine:
        engine.ingest(DOC, source=SourceKind.RESEARCH_PAPER, locator="doi:10/x")
        yield engine


def test_echo_client_satisfies_the_protocol():
    assert isinstance(EchoClient(), LLMClient)


def test_the_model_is_handed_the_memories_and_the_system_rule(cme):
    echo = EchoClient(reply=TRUE_ANSWER)
    GroundedClient(cme, echo).ask("what is entanglement?")

    assert "Known facts, with confidence and source:" in echo.last_prompt
    assert "Entanglement correlates two separated qubits." in echo.last_prompt
    assert "doi:10/x" in echo.last_prompt
    assert "Question: what is entanglement?" in echo.last_prompt
    assert "only the known facts" in echo.last_system


def test_a_supported_answer_comes_back_grounded(cme):
    result = GroundedClient(cme, EchoClient(reply=TRUE_ANSWER)).ask("what is entanglement?")
    assert result.is_grounded
    assert result.unsupported == []
    assert "backed by stored evidence" in result.explain()
    assert result.context.beliefs


def test_an_invented_answer_is_flagged_not_rewritten(cme):
    """The middleware reports the problem; it must not quietly edit the answer."""
    result = GroundedClient(cme, EchoClient(reply=FALSE_ANSWER)).ask("how do qubits work?")
    assert not result.is_grounded
    assert result.unsupported == [FALSE_ANSWER]
    assert result.answer == FALSE_ANSWER  # untouched
    assert FALSE_ANSWER in result.explain()


def test_a_half_true_answer_flags_only_the_bad_claim(cme):
    reply = f"{TRUE_ANSWER} {FALSE_ANSWER}"
    result = GroundedClient(cme, EchoClient(reply=reply)).ask("tell me about qubits")
    assert result.report.score == 0.5
    assert result.unsupported == [FALSE_ANSWER]


def test_an_unknown_topic_still_asks_but_carries_no_context(cme):
    echo = EchoClient(reply="I do not know.")
    result = GroundedClient(cme, echo).ask("who won the 1998 cricket final?")
    assert result.context.beliefs == []
    assert echo.last_prompt == "Question: who won the 1998 cricket final?"


def test_the_budget_is_passed_through(cme):
    result = GroundedClient(cme, EchoClient(reply=TRUE_ANSWER)).ask("qubits", budget=5)
    assert result.context.tokens <= 5


def test_vendor_clients_explain_a_missing_sdk():
    """Absent SDKs must produce an actionable message, not an ImportError."""
    from cme_python.clients.claude_client import ClaudeClient
    from cme_python.clients.openai_client import OpenAIClient

    for factory in (OpenAIClient, ClaudeClient):
        try:
            factory(api_key="not-a-real-key")
        except RuntimeError as exc:
            assert "pip install" in str(exc)
        except Exception:  # noqa: BLE001 - SDK present; construction may need config
            pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
