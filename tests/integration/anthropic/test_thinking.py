"""Anthropic extended thinking (legacy budget_tokens path) — recorded + live.

claude-haiku-4-5 is a <=4.5 generation, so lmux maps ``reasoning_effort`` to the
legacy ``thinking={"type": "enabled", "budget_tokens": N}`` config (4.6+ models
use adaptive thinking instead). ``low`` -> 1024 tokens. Asserts BOTH the request
shape (via ``sent_request``) and that the returned thinking block maps to
``reasoning``. Anthropic reports ``output_tokens_details.thinking_tokens`` folded
into ``output_tokens`` (billed at the output rate); lmux surfaces the thinking
*content* as ``reasoning`` but does not currently map that count to
``Usage.reasoning_tokens`` the way OpenAI does — so cost is the normal input+output.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 2048
_BUDGET_TOKENS = 1024  # reasoning_effort="low" on a <=4.5 model
_ANTHROPIC_VERSION = "2023-06-01"
_PROMPT = "What is 17 * 23? Think step by step, then state the number."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "thinking.json"

# claude-haiku-4-5 published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 1.00 / 1_000_000, "output_rate": 5.00 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-ant-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-ant-mock-not-used"


def _chat(auth: _FakeAuth | None) -> ChatResponse:
    return AnthropicProvider(auth=auth).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, reasoning_effort="low"
    )


class TestThinkingCassette:
    @pytest.mark.integration
    def test_request_shape_and_reasoning(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        sent_request: Callable[..., tuple[dict[str, Any], dict[str, str]]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth())
        # request shape: legacy budget_tokens thinking for a <=4.5 model
        body, _ = sent_request()
        assert body["thinking"] == {"type": "enabled", "budget_tokens": _BUDGET_TOKENS}
        # response: the thinking block maps to reasoning
        assert_chat(resp, provider="anthropic")
        assert resp.reasoning
        assert_cost(resp, **_RATES)


class TestLiveThinking:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_thinking(
        self,
        anthropic_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real ANTHROPIC_API_KEY from env
        assert_chat(resp, provider="anthropic")
        assert resp.reasoning, "extended thinking should populate reasoning"
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, anthropic_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "thinking": {"type": "enabled", "budget_tokens": _BUDGET_TOKENS},
            "messages": [{"role": "user", "content": _PROMPT}],
        }
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert any(b.get("type") == "thinking" for b in data["content"]), "expected a thinking content block"
