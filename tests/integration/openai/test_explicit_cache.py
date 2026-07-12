"""OpenAI explicit prompt caching: a ``CachePointContent`` breakpoint on
gpt-5.6-terra bills a cache write, and cost matches the published rate.
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 512

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "explicit_cache_write.json"

# gpt-5.6-terra published rates ($/token).
_RATES = {
    "input_rate": 2.50 / 1_000_000,
    "output_rate": 15.00 / 1_000_000,
    "cache_read_rate": 0.25 / 1_000_000,
    "cache_write_rate": 3.125 / 1_000_000,
}

# A >1024-token prompt with a unique prefix, so a live/record call is a cold cache write.
_FILLER = " ".join(f"Sentence {i}: the quick brown fox jumps over the lazy dog." for i in range(180))
_PROMPT = f"{uuid.uuid4().hex}\n\n{_FILLER}\n\nReply with exactly the word: pong"


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    messages = [UserMessage(content=[TextContent(text=_PROMPT), CachePointContent()])]
    return OpenAIProvider(auth=auth, transport=transport).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)


@pytest.mark.verified
def test_explicit_cache(
    scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert_chat(resp, provider="openai")
    assert resp.usage is not None
    created = resp.usage.cache_creation_tokens
    assert created is not None
    assert created > 1024  # a real, substantial cache write (the prompt is >1024 tokens by design)
    assert_cost(resp, **_RATES)
