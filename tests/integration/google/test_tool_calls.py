"""Google tool calling on Vertex AI: the model emits a functionCall for a forced tool, and a
follow-up turn carrying the tool result produces a final answer that references it.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import (
    AssistantMessage,
    ChatResponse,
    FunctionCallResult,
    FunctionDefinition,
    Tool,
    ToolCall,
    ToolChoiceFunction,
    ToolMessage,
    UserMessage,
)
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 128
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CALLS_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_tool_calls.json"
_RESULT_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_tool_result.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}

_WEATHER_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    )
)


@pytest.mark.verified
def test_tool_calls(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(
            _MODEL,
            [UserMessage(content="What is the weather in Paris?")],
            tools=[_WEATHER_TOOL],
            tool_choice=ToolChoiceFunction(name="get_weather"),
            max_tokens=_MAX_TOKENS,
            provider_params=_NO_THINK,
        )

    resp = scenario(_CALLS_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert resp.provider == "google"
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].function.name == "get_weather"
    assert "paris" in json.loads(resp.tool_calls[0].function.arguments)["location"].lower()
    assert_cost(resp, **_RATES)


@pytest.mark.verified
def test_tool_result_round_trip(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    history = [
        UserMessage(content="What is the weather in Paris? Use the tool, then tell me in one sentence."),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="call_1", function=FunctionCallResult(name="get_weather", arguments='{"location": "Paris"}')
                )
            ]
        ),
        ToolMessage(content='{"temperature_c": 15, "conditions": "sunny"}', tool_call_id="call_1"),
    ]

    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(
            _MODEL, history, tools=[_WEATHER_TOOL], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
        )

    resp = scenario(_RESULT_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert_chat(resp, provider="google")
    content = (resp.content or "").lower()
    assert "15" in content or "sunny" in content
    assert_cost(resp, **_RATES)
