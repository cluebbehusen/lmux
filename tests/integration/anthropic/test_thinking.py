"""Anthropic extended thinking: a step-by-step prompt populates ``reasoning``, and cost matches the published rate."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 2048
_PROMPT = "What is 17 * 23? Think step by step, then state the number."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "thinking.json"

# claude-haiku-4-5 published rates ($/token).
_RATES = {"input_rate": 1.00 / 1_000_000, "output_rate": 5.00 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, reasoning_effort="low"
    )


@pytest.mark.verified
def test_thinking(
    scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="ANTHROPIC_API_KEY")
    assert_chat(resp, provider="anthropic")
    assert resp.reasoning
    assert_cost(resp, **_RATES)
