"""Claude connector. `pip install anthropic` — not a CME dependency."""

from __future__ import annotations

import os

from cme_python.clients.base import missing_sdk

DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024


class ClaudeClient:
    """Minimal `complete()` over the Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        try:
            from anthropic import Anthropic  # noqa: PLC0415
        except ImportError:
            raise missing_sdk("anthropic", "Claude") from None
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
