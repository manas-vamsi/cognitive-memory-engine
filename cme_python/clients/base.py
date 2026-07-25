"""LLM connectors and the grounding middleware.

The brief's plug-and-play shape: a query goes to the model with CME's memories
attached, and the answer comes back checked against what CME actually knows.
The model is a reasoning client; CME stays the brain.

    [query] -> CME.context -> [model] -> CME.verify -> [answer + proof]

Any object with a `complete(prompt, system=None) -> str` method is a client, so
no SDK is a dependency here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from cme_python.cme import CME, GroundedContext
from cme_python.engines.evidence import GroundingReport


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a prompt into text."""

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


SYSTEM = (
    "Answer using only the known facts provided. "
    "If they do not cover the question, say so plainly rather than guessing."
)


class GroundedAnswer(BaseModel):
    """A model's answer with the memories behind it and a verdict on it."""

    question: str
    answer: str
    context: GroundedContext
    report: GroundingReport

    @property
    def is_grounded(self) -> bool:
        return self.report.is_grounded

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.report.unsupported]

    def explain(self) -> str:
        if self.is_grounded:
            return f"All {len(self.report.checks)} claim(s) backed by stored evidence."
        return "Unbacked claim(s): " + "; ".join(self.unsupported)


class GroundedClient:
    """Wraps any LLM client so its answers arrive with memory and proof.

    Verification is reported, never silently rewritten — the caller decides what
    to do about an unsupported claim. Editing the model's words here would hide
    exactly the failure this layer exists to expose.
    """

    def __init__(self, cme: CME, client: LLMClient, *, system: str = SYSTEM) -> None:
        self.cme = cme
        self.client = client
        self.system = system

    def ask(self, question: str, *, budget: float | None = None) -> GroundedAnswer:
        context = self.cme.context(question, budget=budget)
        answer = self.client.complete(self.build_prompt(question, context), system=self.system)
        return GroundedAnswer(
            question=question,
            answer=answer,
            context=context,
            report=self.cme.verify(answer),
        )

    @staticmethod
    def build_prompt(question: str, context: GroundedContext) -> str:
        block = context.as_prompt()
        return f"{block}\n\nQuestion: {question}" if block else f"Question: {question}"


class EchoClient:
    """Returns whatever it is told to. For tests and offline demos."""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return self.reply or prompt


def missing_sdk(package: str, vendor: str) -> RuntimeError:
    return RuntimeError(
        f"The {vendor} client needs `{package}`, which CME does not install by "
        f"default. Run `pip install {package}`."
    )


def build_client(name: str, model: str = "") -> LLMClient:
    """Construct a connector by name. Raises with the fix if it cannot.

    Kept here rather than in `main` so the API layer holds no vendor knowledge.
    """
    if name == "claude":
        from cme_python.clients.claude_client import ClaudeClient  # noqa: PLC0415

        return ClaudeClient(model=model) if model else ClaudeClient()
    if name == "openai":
        from cme_python.clients.openai_client import OpenAIClient  # noqa: PLC0415

        return OpenAIClient(model=model) if model else OpenAIClient()
    raise ValueError(f"Unknown LLM {name!r}. Set CME_LLM to 'claude' or 'openai'.")
