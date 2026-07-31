"""Gemini thought signatures survive a complete Developer API tool-use loop."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, ToolChoiceFunction, ToolMessage, UserMessage

_MODEL = "gemini-3-flash-preview"
_MAX_TOKENS = 2048
_CALL_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "dev_tool_continuation_call.json"
_RESULT_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "dev_tool_continuation_result.json"
_RATES = {"input_rate": 0.50 / 1_000_000, "output_rate": 3.00 / 1_000_000}

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
def test_tool_continuation_dev(
    scenario: Callable[..., Any],
    dev_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    prompt = UserMessage(content="What is the weather in Denver? Use the tool.")

    def _first(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return dev_provider(auth, transport).chat(
            _MODEL,
            [prompt],
            tools=[_WEATHER_TOOL],
            tool_choice=ToolChoiceFunction(name="get_weather"),
            max_tokens=_MAX_TOKENS,
        )

    first = scenario(_CALL_CASSETTE, _first, requires="GEMINI_API_KEY")
    assert first.tool_calls is not None
    assert first.continuation is not None
    assert_cost(first, **_RATES)

    tool_call = first.tool_calls[0]
    history = [
        prompt,
        first.to_assistant_message(),
        ToolMessage(
            content=json.dumps({"tempF": 68, "conditions": "clear"}),
            tool_call_id=tool_call.id,
        ),
    ]

    def _second(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return dev_provider(auth, transport).chat(
            _MODEL,
            history,
            tools=[_WEATHER_TOOL],
            max_tokens=_MAX_TOKENS,
        )

    second = scenario(_RESULT_CASSETTE, _second, requires="GEMINI_API_KEY")
    assert_chat(second, provider="google")
    assert "68" in (second.content or "") or "clear" in (second.content or "").lower()
    assert_cost(second, **_RATES)
