"""Anthropic streaming cost parity — recorded SSE cassette + live.

Anthropic's stream differs from OpenAI: usage is split across events
(``message_start`` carries input/cache tokens, ``message_delta`` carries output),
which lmux merges into a usage-bearing chunk. A recorded stream and a live stream
must satisfy the same cost contract.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 16
_ANTHROPIC_VERSION = "2023-06-01"
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "chat_stream.json"

# claude-haiku-4-5 published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 1.00 / 1_000_000, "output_rate": 5.00 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-ant-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-ant-mock-not-used"


def _stream(auth: _FakeAuth | None) -> list[ChatChunk]:
    provider = AnthropicProvider(auth=auth)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


def _final_usage_chunk(chunks: list[ChatChunk]) -> ChatChunk:
    """The authoritative (last) usage-bearing chunk — lmux merges message_start +
    message_delta usage onto it."""
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost, "expected at least one usage-bearing chunk"
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
        anthropic_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        chunks = _stream(None)  # real ANTHROPIC_API_KEY from env
        content = "".join(c.delta or "" for c in chunks)
        assert content.strip()
        assert any(c.finish_reason for c in chunks)
        assert_cost(_final_usage_chunk(chunks), **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, anthropic_key: str, record_stream_cassette: Callable[..., str]) -> None:
        body = {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": True,
        }
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        sse = record_stream_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert "message_stop" in sse, "expected a terminated Anthropic event stream"
