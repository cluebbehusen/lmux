"""OpenAI tool calling: a function tool returns a mapped ``get_weather`` tool call
with a location argument, and cost matches the published rate.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 128
_PROMPT = "What is the weather in Paris? Use the get_weather tool."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "tool_calls.json"

# gpt-4o-mini published rates ($/token).
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}

_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    )
)


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return OpenAIProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], tools=[_TOOL], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_tool_calls(scenario: Callable[..., Any], assert_cost: Callable[..., None]) -> None:
    resp = scenario(_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].function.name == "get_weather"
    assert "paris" in json.loads(resp.tool_calls[0].function.arguments)["location"].lower()
    assert_cost(resp, **_RATES)
