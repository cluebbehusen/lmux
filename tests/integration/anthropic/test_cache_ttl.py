"""Anthropic 1h (extended) prompt-cache cost parity — recorded cassettes + live.

The existing suite covers the default 5m TTL; this covers the 1h TTL, which lmux
surfaces as ``cache_creation_tokens_by_ttl={"1h": N}`` and bills at the 1h rate
(2x input, vs the 1.25x default 5m rate). Recorded from live Anthropic
(``claude-haiku-4-5``); a probe confirmed 1h needs no ``anthropic-beta`` header
under anthropic-version 2023-06-01.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 16
_ANTHROPIC_VERSION = "2023-06-01"

_CASSETTES = Path(__file__).parent.parent / "cassettes" / "anthropic"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_1h_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_1h_read.json"

# claude-haiku-4-5 published rates ($/token) — the independent source of truth.
_INPUT_RATE = 1.00 / 1_000_000
_OUTPUT_RATE = 5.00 / 1_000_000
_CACHE_READ_RATE = 0.10 / 1_000_000
_CACHE_WRITE_1H_RATE = 2.00 / 1_000_000


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-ant-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-ant-mock-not-used"


def _cached_1h(prompt: str) -> list[UserMessage]:
    return [UserMessage(content=[TextContent(text=prompt), CachePointContent(ttl="1h")])]


def _chat(auth: _FakeAuth | None, messages: list[UserMessage]) -> ChatResponse:
    return AnthropicProvider(auth=auth).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)


class TestCache1hWriteCassette:
    @pytest.mark.integration
    def test_1h_write_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_WRITE_CASSETTE)
        resp = _chat(_FakeAuth(), _cached_1h("replayed"))
        assert_chat(resp, provider="anthropic")
        assert resp.usage is not None
        assert resp.usage.cache_creation_tokens_by_ttl is not None
        assert set(resp.usage.cache_creation_tokens_by_ttl) == {"1h"}, "expected a 1h-only cache write"
        assert_cost(
            resp,
            input_rate=_INPUT_RATE,
            output_rate=_OUTPUT_RATE,
            cache_read_rate=_CACHE_READ_RATE,
            cache_write_rate_by_ttl={"1h": _CACHE_WRITE_1H_RATE},
        )


class TestCache1hReadCassette:
    @pytest.mark.integration
    def test_1h_read_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_READ_CASSETTE)
        resp = _chat(_FakeAuth(), _cached_1h("replayed"))
        assert_chat(resp, provider="anthropic")
        assert resp.usage is not None
        assert resp.usage.cache_read_tokens
        assert resp.usage.cache_creation_tokens is None
        assert_cost(resp, input_rate=_INPUT_RATE, output_rate=_OUTPUT_RATE, cache_read_rate=_CACHE_READ_RATE)


class TestLiveCache1hParity:
    @pytest.mark.integration
    @pytest.mark.live
    def test_cold_write_then_warm_read(
        self,
        anthropic_key: str,  # noqa: ARG002 — requested to skip when unset
        cache_prompt: Callable[..., str],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        messages = _cached_1h(cache_prompt(300))

        write = _chat(None, messages)
        assert_chat(write, provider="anthropic")
        assert write.usage is not None
        assert write.usage.cache_creation_tokens_by_ttl is not None
        assert "1h" in write.usage.cache_creation_tokens_by_ttl
        assert_cost(
            write,
            input_rate=_INPUT_RATE,
            output_rate=_OUTPUT_RATE,
            cache_read_rate=_CACHE_READ_RATE,
            cache_write_rate_by_ttl={"1h": _CACHE_WRITE_1H_RATE},
        )

        read = _chat(None, messages)
        assert_chat(read, provider="anthropic")
        assert read.usage is not None
        assert read.usage.cache_read_tokens, "warm call should report a cache read"
        assert_cost(read, input_rate=_INPUT_RATE, output_rate=_OUTPUT_RATE, cache_read_rate=_CACHE_READ_RATE)


class TestRecordCassettes:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(
        self, anthropic_key: str, cache_prompt: Callable[..., str], record_cassette: Callable[..., dict[str, Any]]
    ) -> None:
        prompt = cache_prompt(300)
        body = {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                }
            ],
        }
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        cold = record_cassette(_WRITE_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert cold["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] > 0, "expected a 1h cache write"
        warm = record_cassette(_READ_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert warm["usage"]["cache_read_input_tokens"] > 0, "expected a warm cache read"
