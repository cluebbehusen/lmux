"""OpenAI reasoning chat: the response reports reasoning tokens as a subset of
output tokens, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 384
_PROMPT = "What is 17 * 23? Reason step by step, then state the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "reasoning.json"

# gpt-5.6-terra published rates ($/token).
_RATES = {
    "input_rate": 2.50 / 1_000_000,
    "output_rate": 15.00 / 1_000_000,
    "cache_read_rate": 0.25 / 1_000_000,
    "cache_write_rate": 3.125 / 1_000_000,
}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return OpenAIProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_reasoning(
    scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert_chat(resp, provider="openai")
    assert "391" in resp.content  # the reasoning produced the correct answer (17 * 23)
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens
    assert resp.usage.reasoning_tokens <= resp.usage.output_tokens
    assert_cost(resp, **_RATES)
