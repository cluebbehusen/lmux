"""OpenAI tool calling — recorded + live.

A chat with a function tool returns ``tool_calls`` (and ``content`` is None), so
``assert_chat`` doesn't apply — this asserts the mapped tool call and the cost.
``gpt-4o-mini`` keeps it cheap.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 128
_PROMPT = "What is the weather in Paris? Use the get_weather tool."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "tool_calls.json"

# gpt-4o-mini published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}

_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    )
)
_TOOL_WIRE = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    },
}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _chat(auth: _FakeAuth | None) -> ChatResponse:
    return OpenAIProvider(auth=auth).chat(_MODEL, [UserMessage(content=_PROMPT)], tools=[_TOOL], max_tokens=_MAX_TOKENS)


class TestToolCallCassette:
    @pytest.mark.integration
    def test_tool_call_mapping_and_cost(
        self, mount_cassette: Callable[[Path], dict[str, Any]], assert_cost: Callable[..., None]
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth())
        assert resp.finish_reason == "tool_calls"
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].function.name == "get_weather"
        assert json.loads(resp.tool_calls[0].function.arguments)["location"]  # valid JSON args with a location
        assert_cost(resp, **_RATES)


class TestLiveToolCall:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_tool_call(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real OPENAI_API_KEY from env
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].function.name == "get_weather"
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "tools": [_TOOL_WIRE],
            "tool_choice": "auto",
            "max_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert data["choices"][0]["message"]["tool_calls"], "expected a tool call"
