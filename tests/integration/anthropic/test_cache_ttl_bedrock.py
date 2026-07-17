"""Native Anthropic-on-Bedrock 1h (extended) prompt-cache cost: a cold call writes 1h-TTL
cache tokens billed at the 1h rate, and an identical warm call reports a cache read.

Selecting a cache TTL is native-API-only — Bedrock's Converse ``cachePoint`` writes at the
default 5m TTL with no way to request 1h — so this is one of the capabilities the native
transport exists to provide.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 16
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "anthropic"
_WRITE_CASSETTE = _CASSETTES / "bedrock_cache_1h_write.json"
_READ_CASSETTE = _CASSETTES / "bedrock_cache_1h_read.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_INPUT_RATE = 1.10 / 1_000_000
_OUTPUT_RATE = 5.50 / 1_000_000
_CACHE_READ_RATE = 0.11 / 1_000_000
_CACHE_WRITE_1H_RATE = 2.20 / 1_000_000


@pytest.mark.verified
def test_cold_write_then_warm_read(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    cache_prompt: Callable[..., str],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    messages = [UserMessage(content=[TextContent(text=cache_prompt(300)), CachePointContent(ttl="1h")])]

    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
            _MODEL, messages, max_tokens=_MAX_TOKENS
        )

    write = scenario(_WRITE_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(write, provider="anthropic-bedrock")
    assert write.usage is not None
    created = write.usage.cache_creation_tokens
    assert created is not None
    assert created > 1024  # a real, substantial cache write (the prompt is >1024 tokens by design)
    assert write.usage.cache_creation_tokens_by_ttl == {"1h": created}  # all of it written at the 1h TTL
    assert write.usage.cache_read_tokens is None  # a cold write reads nothing
    assert_cost(
        write,
        input_rate=_INPUT_RATE,
        output_rate=_OUTPUT_RATE,
        cache_read_rate=_CACHE_READ_RATE,
        cache_write_rate_by_ttl={"1h": _CACHE_WRITE_1H_RATE},
    )

    read = scenario(_READ_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(read, provider="anthropic-bedrock")
    assert read.usage is not None
    assert read.usage.cache_creation_tokens is None  # a warm read writes nothing new
    assert read.usage.cache_read_tokens == created  # reads back exactly the tokens the 1h write created
    assert_cost(read, input_rate=_INPUT_RATE, output_rate=_OUTPUT_RATE, cache_read_rate=_CACHE_READ_RATE)
