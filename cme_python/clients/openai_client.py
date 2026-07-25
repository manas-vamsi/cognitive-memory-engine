"""OpenAI connector. `pip install openai` — not a CME dependency."""

from __future__ import annotations

import os

from cme_python.clients.base import missing_sdk

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient:
    """Minimal `complete()` over the Chat Completions API."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            raise missing_sdk("openai", "OpenAI") from None
        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""
