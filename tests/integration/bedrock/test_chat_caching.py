"""AWS Bedrock prompt caching (Converse cachePoint, Anthropic Claude): a cold call
writes cache tokens and an identical warm call reads them back, each billed at the rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 16
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "bedrock"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_read.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {
    "input_rate": 1.10 / 1_000_000,
    "output_rate": 5.50 / 1_000_000,
    "cache_read_rate": 0.11 / 1_000_000,
    "cache_write_rate": 1.375 / 1_000_000,
}


@pytest.mark.verified
def test_cold_write_then_warm_read(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    cache_prompt: Callable[..., str],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    messages = [UserMessage(content=[TextContent(text=cache_prompt(300)), CachePointContent()])]

    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return BedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
            _MODEL, messages, max_tokens=_MAX_TOKENS
        )

    write = scenario(_WRITE_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    assert_chat(write, provider="aws-bedrock")
    assert write.usage is not None
    created = write.usage.cache_creation_tokens
    assert created is not None
    assert created > 1024  # a real, substantial cache write (the prompt is >1024 tokens by design)
    assert write.usage.cache_read_tokens is None  # a cold write reads nothing
    assert_cost(write, **_RATES)

    read = scenario(_READ_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    assert_chat(read, provider="aws-bedrock")
    assert read.usage is not None
    assert read.usage.cache_creation_tokens is None  # a warm read writes nothing new
    assert read.usage.cache_read_tokens == created  # reads back exactly the tokens the write created
    assert_cost(read, **_RATES)
