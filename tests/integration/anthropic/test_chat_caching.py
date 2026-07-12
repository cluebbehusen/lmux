"""Anthropic prompt-cache cost (default 5m TTL): a cold call reports a cache write
and an identical warm call reports a cache read, each billed at the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 16
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "anthropic"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_read.json"

# claude-haiku-4-5 published rates ($/token).
_RATES = {
    "input_rate": 1.00 / 1_000_000,
    "output_rate": 5.00 / 1_000_000,
    "cache_read_rate": 0.10 / 1_000_000,
    "cache_write_rate": 1.25 / 1_000_000,
}


@pytest.mark.verified
def test_cold_write_then_warm_read(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    cache_prompt: Callable[..., str],
) -> None:
    messages = [UserMessage(content=[TextContent(text=cache_prompt(280)), CachePointContent()])]

    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return AnthropicProvider(auth=auth, transport=transport).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)

    write = scenario(_WRITE_CASSETTE, _chat, requires="ANTHROPIC_API_KEY")
    assert_chat(write, provider="anthropic")
    assert write.usage is not None
    created = write.usage.cache_creation_tokens
    assert created is not None
    assert created > 1024  # a real, substantial cache write (the prompt is >1024 tokens by design)
    assert write.usage.cache_read_tokens is None  # a cold write reads nothing
    assert_cost(write, **_RATES)

    read = scenario(_READ_CASSETTE, _chat, requires="ANTHROPIC_API_KEY")
    assert_chat(read, provider="anthropic")
    assert read.usage is not None
    assert read.usage.cache_creation_tokens is None  # a warm read writes nothing new
    assert read.usage.cache_read_tokens == created  # reads back exactly the tokens the write created
    assert_cost(read, **_RATES)
