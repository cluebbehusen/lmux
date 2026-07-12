"""OpenAI reasoning-token usage (gpt-5.6+) — recorded + live.

gpt-5.6 reports ``completion_tokens_details.reasoning_tokens``, which lmux
surfaces as ``Usage.reasoning_tokens`` (a subset of ``output_tokens``, billed at
the output rate). Proves the reasoning-token mapping and that cost is unaffected
by it (reasoning tokens are already counted in output_tokens).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-5.6-terra"
_MAX_TOKENS = 384
_PROMPT = "What is 17 * 23? Reason step by step, then state the number."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "reasoning.json"

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


def _chat(auth: _FakeAuth | None) -> ChatResponse:
    return OpenAIProvider(auth=auth).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS)


class TestReasoningCassette:
    @pytest.mark.integration
    def test_reasoning_tokens_and_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth())
        assert_chat(resp, provider="openai")
        assert resp.usage is not None
        assert resp.usage.reasoning_tokens
        assert resp.usage.reasoning_tokens <= resp.usage.output_tokens  # reasoning is a subset of output
        assert_cost(resp, **_RATES)


class TestLiveReasoning:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_reasoning(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real OPENAI_API_KEY from env
        assert_chat(resp, provider="openai")
        assert resp.usage is not None
        assert resp.usage.reasoning_tokens, "gpt-5.6 should report reasoning tokens"
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_completion_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, "expected reasoning tokens"
