"""OpenAI explicit prompt caching (gpt-5.6+) — recorded cassette + live parity.

Guards the exact wire contract that regressed twice in review: a
``CachePointContent`` becomes a ``prompt_cache_breakpoint`` on the preceding
content block plus a root ``prompt_cache_options``, and gpt-5.6 bills the
resulting cache write. Asserts BOTH the outgoing request shape (``sent_request``)
and the cache-write cost, from a real recording.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ChatResponse, TextContent, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 512

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "explicit_cache_write.json"

# gpt-5.6-terra tier-1 published rates ($/token) — the independent source of truth.
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


def _cached(prompt: str) -> list[UserMessage]:
    return [UserMessage(content=[TextContent(text=prompt), CachePointContent()])]


def _chat(auth: _FakeAuth | None, messages: list[UserMessage]) -> ChatResponse:
    return OpenAIProvider(auth=auth).chat(_MODEL, messages, max_tokens=_MAX_TOKENS)


class TestExplicitCacheWrite:
    @pytest.mark.integration
    def test_request_shape_and_write_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        sent_request: Callable[..., tuple[dict[str, Any], dict[str, str]]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth(), _cached("replayed prefix"))
        # request shape: breakpoint on the preceding block + root option
        body, _ = sent_request()
        assert body["prompt_cache_options"] == {"mode": "explicit"}
        assert body["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        # write cost
        assert_chat(resp, provider="openai")
        assert resp.usage is not None
        assert resp.usage.cache_creation_tokens
        assert_cost(resp, **_RATES)


class TestLiveExplicitCache:
    @pytest.mark.integration
    @pytest.mark.live
    def test_cold_write(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        cache_prompt: Callable[..., str],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        write = _chat(None, _cached(cache_prompt()))
        assert_chat(write, provider="openai")
        assert write.usage is not None
        assert write.usage.cache_creation_tokens, "cold call should report a cache write"
        assert_cost(write, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(
        self, openai_key: str, cache_prompt: Callable[..., str], record_cassette: Callable[..., dict[str, Any]]
    ) -> None:
        prompt = cache_prompt()
        body = {
            "model": _MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt, "prompt_cache_breakpoint": {"mode": "explicit"}}],
                }
            ],
            "prompt_cache_options": {"mode": "explicit"},
            "max_completion_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        cold = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert cold["usage"]["prompt_tokens_details"]["cache_write_tokens"] > 0, "expected a cold cache write"
