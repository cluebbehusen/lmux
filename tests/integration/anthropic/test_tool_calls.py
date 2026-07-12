"""Anthropic tool calling — recorded + live.

A chat with a tool returns a ``tool_use`` content block, which lmux maps to
``tool_calls`` (and ``stop_reason: tool_use`` -> ``finish_reason: tool_calls``);
assert the mapped call + cost. ``claude-haiku-4-5``.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 256
_ANTHROPIC_VERSION = "2023-06-01"
_PROMPT = "What is the weather in Paris? Use the get_weather tool."
_PARAMS = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}

_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather", description="Get the current weather for a city.", parameters=_PARAMS
    )
)
_TOOL_WIRE = {"name": "get_weather", "description": "Get the current weather for a city.", "input_schema": _PARAMS}

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "tool_calls.json"

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
        _MODEL, [UserMessage(content=_PROMPT)], tools=[_TOOL], max_tokens=_MAX_TOKENS
    )


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
        anthropic_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real ANTHROPIC_API_KEY from env
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].function.name == "get_weather"
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, anthropic_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": _PROMPT}],
            "tools": [_TOOL_WIRE],
            "tool_choice": {"type": "auto"},
        }
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert any(b.get("type") == "tool_use" for b in data["content"]), "expected a tool_use content block"
