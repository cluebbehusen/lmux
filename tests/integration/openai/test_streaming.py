"""OpenAI streaming cost parity — SSE cassette + the usage-bearing final chunk.

lmux requests ``stream_options.include_usage``, so the terminal chunk carries
usage (and thus ``.cost``). This proves the harness handles a non-JSON (SSE)
cassette shape and that streaming cost math matches a recorded and a live run.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 32
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "chat_stream.json"

# gpt-4o-mini published rates ($ per token) — independent source of truth.
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _stream(auth: _FakeAuth | None) -> list[ChatChunk]:
    provider = OpenAIProvider(auth=auth)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


def _usage_chunk(chunks: list[ChatChunk]) -> ChatChunk:
    """The single terminal chunk that carries usage (and cost)."""
    with_cost = [c for c in chunks if c.cost is not None]
    assert len(with_cost) == 1, f"expected exactly one usage-bearing chunk, got {len(with_cost)}"
    return with_cost[0]


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
        assert_cost(_usage_chunk(chunks), **_RATES)


class TestLiveStream:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_stream(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        chunks = _stream(None)  # real OPENAI_API_KEY from env
        content = "".join(c.delta or "" for c in chunks)
        assert content.strip()
        assert any(c.finish_reason for c in chunks)
        assert_cost(_usage_chunk(chunks), **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_stream_cassette: Callable[..., str]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": _MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        sse = record_stream_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert "[DONE]" in sse, "expected a terminated SSE stream"
