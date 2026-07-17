"""Claude on Bedrock via the native Anthropic Messages API (InvokeModel): a Bedrock
inference-profile ID returns the expected word, and cost matches the published rate.

Authenticates with the AWS_BEARER_TOKEN_BEDROCK API key (the SigV4 path is covered by
test_sigv4_auth_bedrock). The response echoes a region-less model ID
(``claude-haiku-4-5-20251001``), so pricing must resolve from the region-prefixed request ID.
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
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "bedrock_chat.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_chat_bedrock(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(resp, provider="anthropic-bedrock")
    assert "pong" in (resp.content or "").lower()
    assert resp.model == "claude-haiku-4-5-20251001"  # the native response echoes the region-less ID
    assert_cost(resp, **_RATES)
