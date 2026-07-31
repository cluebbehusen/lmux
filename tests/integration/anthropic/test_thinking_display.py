"""Claude adaptive thinking returns its summarized display through lmux."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import AnthropicProvider

_MODEL = "claude-sonnet-5"
_MAX_TOKENS = 2048
_PROMPT = "Think it through step by step: which is larger, 17/23 or 31/42? Show your reasoning."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "thinking_display.json"

_RATES = {"input_rate": 2.00 / 1_000_000, "output_rate": 10.00 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicProvider(auth=auth, transport=transport).chat(
        _MODEL,
        [UserMessage(content=_PROMPT)],
        max_tokens=_MAX_TOKENS,
        reasoning_effort="high",
        provider_params=AnthropicParams(pricing_as_of=date(2026, 7, 31)),
    )


@pytest.mark.verified
def test_thinking_display(
    scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="ANTHROPIC_API_KEY")
    assert_chat(resp, provider="anthropic")
    assert resp.reasoning
    assert "17/23" in resp.content
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens is not None
    assert resp.usage.reasoning_tokens > 0
    assert_cost(resp, **_RATES)
