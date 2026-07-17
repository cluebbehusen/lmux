"""Native Anthropic-on-Bedrock extended thinking: a step-by-step prompt populates
``reasoning``, and cost matches the published rate.

Bedrock's native passthrough omits the ``usage.output_tokens_details`` breakdown the direct
Anthropic API returns, so ``usage.reasoning_tokens`` is None here — unlike the direct-API
thinking test, which asserts it is populated.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 2048
_PROMPT = "What is 17 * 23? Think step by step, then state the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "bedrock_thinking.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, reasoning_effort="low"
    )


@pytest.mark.verified
def test_thinking_bedrock(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(resp, provider="anthropic-bedrock")
    assert resp.reasoning  # the thinking block round-tripped through Bedrock's passthrough
    assert "391" in resp.content  # the thinking produced the correct answer (17 * 23)
    assert_cost(resp, **_RATES)
