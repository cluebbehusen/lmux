"""OpenAI prompt-cache cost parity: a recorded cassette and the live endpoint
must satisfy the *same* cost-math contract for cache writes and cache reads.

This is the moto idea adapted for a cost-reporting proxy: we can't assert the
model's output is identical, but we can assert the money is. ``gpt-5.6-terra``
bills cache writes (older models write for free), so it exercises both the
cache-creation and cache-read cost paths.

The harness (``conftest.py``) supplies the reusable pieces — this module only
declares the endpoint, model, cassette paths, and terra's published rates.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 512

_CASSETTES = Path(__file__).parent.parent / "cassettes" / "openai"
_WRITE_CASSETTE = _CASSETTES / "chat_cache_write.json"
_READ_CASSETTE = _CASSETTES / "chat_cache_read.json"

# gpt-5.6-terra tier-1 published rates ($ per token) — the independent source of
# truth for the cost math (see conftest ``assert_cost``).
_RATES = {
    "input_rate": 2.50 / 1_000_000,
    "output_rate": 15.00 / 1_000_000,
    "cache_read_rate": 0.25 / 1_000_000,
    "cache_write_rate": 3.125 / 1_000_000,
}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _chat(auth: _FakeAuth | None, messages: list[UserMessage]) -> ChatResponse:
    return OpenAIProvider(auth=auth).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)


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
        resp = _chat(_FakeAuth(), [UserMessage(content="replayed")])
        assert_chat(resp, provider="openai")
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
        resp = _chat(_FakeAuth(), [UserMessage(content="replayed")])
        assert_chat(resp, provider="openai")
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
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        cache_prompt: Callable[[], str],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        messages = [UserMessage(content=cache_prompt())]

        write = _chat(None, messages)
        assert_chat(write, provider="openai")
        assert write.usage is not None
        assert write.usage.cache_creation_tokens, "cold call should report a cache write"
        assert_cost(write, **_RATES)

        read = _chat(None, messages)
        assert_chat(read, provider="openai")
        assert read.usage is not None
        assert read.usage.cache_read_tokens, "warm call should report a cache read"
        assert_cost(read, **_RATES)


# MARK: Recorder


class TestRecordCassettes:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(
        self, openai_key: str, cache_prompt: Callable[[], str], record_cassette: Callable[..., dict[str, Any]]
    ) -> None:
        prompt = cache_prompt()
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        cold = record_cassette(_WRITE_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert cold["usage"]["prompt_tokens_details"]["cache_write_tokens"] > 0, "expected a cold cache write"
        warm = record_cassette(_READ_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "expected a warm cache read"
