"""Native Anthropic-on-Bedrock streaming (InvokeModelWithResponseStream): the streamed
content includes the expected word, and the terminal usage-bearing chunk's cost matches the rate.

Bedrock wraps each native Anthropic streaming event in an AWS binary event-stream frame (not
SSE), so this exercises frame decoding plus base64 unwrapping — recorded verbatim as raw bytes.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 32
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "bedrock_stream.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401 — harness-supplied per mode
    provider = AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


@pytest.mark.verified
def test_streaming_bedrock(
    scenario: Callable[..., Any],
    assert_cost: Callable[..., None],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    chunks = scenario(_CASSETTE, _stream, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    content = "".join(c.delta or "" for c in chunks)
    assert "pong" in content.lower()
    assert any(c.finish_reason for c in chunks)
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost
    assert with_cost[-1].provider == "anthropic-bedrock"
    assert_cost(with_cost[-1], **_RATES)
