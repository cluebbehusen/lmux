"""Anthropic signed thinking survives a complete tool-use loop."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, ToolMessage, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 2048
_CALL_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "tool_continuation_call.json"
_RESULT_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "tool_continuation_result.json"

_WEATHER_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)


@pytest.mark.verified
def test_tool_continuation(scenario: Callable[..., Any]) -> None:
    prompt = UserMessage(content="What is the weather in Denver? Use the get_weather tool.")

    def _first(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
        return AnthropicProvider(auth=auth, transport=transport).chat(
            _MODEL,
            [prompt],
            tools=[_WEATHER_TOOL],
            max_tokens=_MAX_TOKENS,
            reasoning_effort="low",
        )

    first = scenario(_CALL_CASSETTE, _first, requires="ANTHROPIC_API_KEY")
    assert first.tool_calls is not None
    assert first.continuation is not None

    tool_call = first.tool_calls[0]
    history = [
        prompt,
        first.to_assistant_message(),
        ToolMessage(content=json.dumps({"tempF": 68, "conditions": "clear"}), tool_call_id=tool_call.id),
    ]

    def _second(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
        return AnthropicProvider(auth=auth, transport=transport).chat(
            _MODEL,
            history,
            tools=[_WEATHER_TOOL],
            max_tokens=_MAX_TOKENS,
            reasoning_effort="low",
        )

    second = scenario(_RESULT_CASSETTE, _second, requires="ANTHROPIC_API_KEY")
    assert "68" in (second.content or "") or "clear" in (second.content or "").lower()
