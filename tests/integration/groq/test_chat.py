"""Groq chat cost parity — a recorded cassette and the live endpoint satisfy the
same cost contract, proving the math holds for a third (OpenAI-wire-compatible)
provider. ``llama-3.1-8b-instant`` bills input+output only; a short prompt and a
small ``max_tokens`` keep the recording cheap.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_groq.provider import GroqProvider

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "groq" / "chat.json"

# llama-3.1-8b-instant published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 0.05 / 1_000_000, "output_rate": 0.08 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "gsk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "gsk-mock-not-used"


def _chat(auth: _FakeAuth | None) -> ChatResponse:
    return GroqProvider(auth=auth).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS)


class TestChatCassette:
    @pytest.mark.integration
    def test_chat_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth())
        assert_chat(resp, provider="groq")
        assert_cost(resp, **_RATES)


class TestLiveChat:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_chat(
        self,
        groq_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real GROQ_API_KEY from env
        assert_chat(resp, provider="groq")
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, groq_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {groq_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert data["usage"]["prompt_tokens"] > 0, "expected input token usage"
