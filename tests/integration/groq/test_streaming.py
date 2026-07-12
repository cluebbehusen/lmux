"""Groq streaming cost parity — SSE cassette + the usage-bearing final chunk.

Groq requests ``stream_options.include_usage`` (OpenAI-compatible), so the
terminal chunk carries usage and thus ``.cost``; a recorded stream and a live
stream must satisfy the same cost contract.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_groq.provider import GroqProvider

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "groq" / "chat_stream.json"

# llama-3.1-8b-instant published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 0.05 / 1_000_000, "output_rate": 0.08 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "gsk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "gsk-mock-not-used"


def _stream(auth: _FakeAuth | None) -> list[ChatChunk]:
    provider = GroqProvider(auth=auth)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


def _final_usage_chunk(chunks: list[ChatChunk]) -> ChatChunk:
    """The authoritative (last) usage-bearing chunk.

    Unlike OpenAI (one trailing usage chunk), Groq repeats usage on *both* the
    finish chunk and a trailing chunk — a real wire difference this recorded
    cassette surfaced. All usage-bearing chunks agree; the last is authoritative
    (a consumer must take the terminal usage, not sum across chunks)."""
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost, "expected at least one usage-bearing chunk"
    assert all(c.usage == with_cost[-1].usage for c in with_cost), "repeated usage chunks must agree"
    return with_cost[-1]


class TestStreamCassette:
    @pytest.mark.integration
    def test_stream_cost(
        self, mount_cassette: Callable[[Path], dict[str, Any]], assert_cost: Callable[..., None]
    ) -> None:
        mount_cassette(_CASSETTE)
        chunks = _stream(_FakeAuth())
        content = "".join(c.delta or "" for c in chunks)
        assert content.strip()
        assert any(c.finish_reason for c in chunks)
        assert_cost(_final_usage_chunk(chunks), **_RATES)


class TestLiveStream:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_stream(
        self,
        groq_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        chunks = _stream(None)  # real GROQ_API_KEY from env
        content = "".join(c.delta or "" for c in chunks)
        assert content.strip()
        assert any(c.finish_reason for c in chunks)
        assert_cost(_final_usage_chunk(chunks), **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, groq_key: str, record_stream_cassette: Callable[..., str]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": _MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {groq_key}"}
        sse = record_stream_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert "[DONE]" in sse, "expected a terminated SSE stream"
