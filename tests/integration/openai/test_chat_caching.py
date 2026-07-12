"""OpenAI prompt-cache cost: a cold call reports a cache write and an identical
warm call reports a cache read, each billed at the published rate. gpt-5.6-terra
bills cache writes (older models write for free).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 512
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "openai"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_read.json"

# gpt-5.6-terra tier-1 published rates ($/token).
_RATES = {
    "input_rate": 2.50 / 1_000_000,
    "output_rate": 15.00 / 1_000_000,
    "cache_read_rate": 0.25 / 1_000_000,
    "cache_write_rate": 3.125 / 1_000_000,
}


@pytest.mark.verified
def test_cold_write_then_warm_read(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    cache_prompt: Callable[..., str],
) -> None:
    messages = [UserMessage(content=cache_prompt())]  # >1024-token unique prompt, reused for both calls

    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return OpenAIProvider(auth=auth, transport=transport).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)

    write = scenario(_WRITE_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert_chat(write, provider="openai")
    assert write.usage is not None
    assert write.usage.cache_creation_tokens
    assert write.usage.cache_read_tokens is None
    assert_cost(write, **_RATES)

    read = scenario(_READ_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert_chat(read, provider="openai")
    assert read.usage is not None
    assert read.usage.cache_read_tokens
    assert read.usage.cache_creation_tokens is None
    assert_cost(read, **_RATES)
