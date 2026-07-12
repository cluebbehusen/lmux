"""Groq chat: a deterministic prompt returns the expected word, and cost matches
the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_groq.provider import GroqProvider

_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "groq" / "chat.json"

# llama-3.1-8b-instant published rates ($/token).
_RATES = {"input_rate": 0.05 / 1_000_000, "output_rate": 0.08 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return GroqProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_chat(scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]) -> None:
    resp = scenario(_CASSETTE, _chat, requires="GROQ_API_KEY")
    assert_chat(resp, provider="groq")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
