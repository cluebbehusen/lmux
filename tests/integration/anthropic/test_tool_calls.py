"""Anthropic tool calling: a weather prompt returns a get_weather tool call with a
JSON location argument, and cost matches the published rate.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, UserMessage
from lmux_anthropic.provider import AnthropicProvider

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 256
_PROMPT = "What is the weather in Paris? Use the get_weather tool."
_PARAMS = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}

_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather", description="Get the current weather for a city.", parameters=_PARAMS
    )
)

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "tool_calls.json"

# claude-haiku-4-5 published rates ($/token).
_RATES = {"input_rate": 1.00 / 1_000_000, "output_rate": 5.00 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], tools=[_TOOL], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_tool_calls(scenario: Callable[..., Any], assert_cost: Callable[..., None]) -> None:
    resp = scenario(_CASSETTE, _chat, requires="ANTHROPIC_API_KEY")
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].function.name == "get_weather"
    assert json.loads(resp.tool_calls[0].function.arguments)["location"]
    assert_cost(resp, **_RATES)
