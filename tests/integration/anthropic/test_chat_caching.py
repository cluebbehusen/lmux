"""Anthropic prompt-cache cost parity — a cross-provider mirror of the OpenAI
terra scenario, proving the same cost-math contract holds for a second provider.

``claude-haiku-4-5`` has flat (non-scheduled) pricing and explicit
``cache_control`` breakpoints, exercising the cache-creation and cache-read cost
paths. Anthropic reports ``input_tokens`` exclusive of cached tokens; lmux
normalizes to the total, so the same generic cost formula applies.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_BASE_URL = "https://api.anthropic.com"
_ENDPOINT = f"{_BASE_URL}/v1/messages"
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 16
_ANTHROPIC_VERSION = "2023-06-01"

_CASSETTES = Path(__file__).parent.parent / "cassettes" / "anthropic"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_read.json"

# claude-haiku-4-5 published rates ($ per token) — the independent source of
# truth for the cost math (5m cache writes bill at the default creation rate).
_RATES = {
    "input_rate": 1.00 / 1_000_000,
    "output_rate": 5.00 / 1_000_000,
    "cache_read_rate": 0.10 / 1_000_000,
    "cache_write_rate": 1.25 / 1_000_000,
}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-ant-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-ant-mock-not-used"


def _cached_messages(prompt: str) -> list[UserMessage]:
    """A user message whose large prefix ends in a cache breakpoint."""
    return [UserMessage(content=[TextContent(text=prompt), CachePointContent()])]


def _chat(auth: _FakeAuth | None, messages: list[UserMessage]) -> ChatResponse:
    return AnthropicProvider(auth=auth).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)


# MARK: Cassette (offline) cost-math


class TestCacheWriteCassette:
    @pytest.mark.integration
    def test_cache_write_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_WRITE_CASSETTE)
        resp = _chat(_FakeAuth(), _cached_messages("replayed"))
        assert_chat(resp, provider="anthropic")
        assert resp.usage is not None
        assert resp.usage.cache_creation_tokens
        assert resp.usage.cache_read_tokens is None
        assert_cost(resp, **_RATES)


class TestCacheReadCassette:
    @pytest.mark.integration
    def test_cache_read_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_READ_CASSETTE)
        resp = _chat(_FakeAuth(), _cached_messages("replayed"))
        assert_chat(resp, provider="anthropic")
        assert resp.usage is not None
        assert resp.usage.cache_read_tokens
        assert resp.usage.cache_creation_tokens is None
        assert_cost(resp, **_RATES)


# MARK: Live parity


class TestLiveCacheParity:
    @pytest.mark.integration
    @pytest.mark.live
    def test_cold_write_then_warm_read(
        self,
        anthropic_key: str,  # noqa: ARG002 — requested to skip when unset
        cache_prompt: Callable[..., str],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        messages = _cached_messages(cache_prompt(280))

        write = _chat(None, messages)
        assert_chat(write, provider="anthropic")
        assert write.usage is not None
        assert write.usage.cache_creation_tokens, "cold call should report a cache write"
        assert_cost(write, **_RATES)

        read = _chat(None, messages)
        assert_chat(read, provider="anthropic")
        assert read.usage is not None
        assert read.usage.cache_read_tokens, "warm call should report a cache read"
        assert_cost(read, **_RATES)


# MARK: Recorder


class TestRecordCassettes:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(
        self, anthropic_key: str, cache_prompt: Callable[..., str], record_cassette: Callable[..., dict[str, Any]]
    ) -> None:
        prompt = cache_prompt(280)
        body = {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}
            ],
        }
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        cold = record_cassette(_WRITE_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert cold["usage"]["cache_creation_input_tokens"] > 0, "expected a cold cache write"
        warm = record_cassette(_READ_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert warm["usage"]["cache_read_input_tokens"] > 0, "expected a warm cache read"
