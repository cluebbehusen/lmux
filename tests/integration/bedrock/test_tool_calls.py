"""AWS Bedrock tool calling (Converse toolConfig, Anthropic Claude): a weather prompt
returns a mapped get_weather tool call with a Paris location, and cost matches the rate.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Tool, UserMessage
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 256
_PROMPT = "What is the weather in Paris? Use the get_weather tool."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "tool_calls.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}

_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    )
)


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return BedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], tools=[_TOOL], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_tool_calls(
    scenario: Callable[..., Any],
    assert_cost: Callable[..., None],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].function.name == "get_weather"
    assert "paris" in json.loads(resp.tool_calls[0].function.arguments)["location"].lower()
    assert_cost(resp, **_RATES)
