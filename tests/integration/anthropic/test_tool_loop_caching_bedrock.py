"""Native Anthropic-on-Bedrock prompt caching across the hops of an agentic tool loop: a cache
point placed after the tool results caches the accumulated tool history, not just the question.

This transport reuses the Anthropic Messages mapper but bills at Bedrock's rates, so it is the
path most Claude-on-Bedrock callers are on and neither the direct-API nor the Converse suite
covers it. Three hops are the fewest that show the behaviour: two cannot distinguish caching
the question from caching the tool history, while the third reads a prefix the previous hop's
tool result extended.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import (
    CachePointContent,
    ChatResponse,
    FunctionDefinition,
    Message,
    TextContent,
    Tool,
    ToolMessage,
    UserMessage,
)
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 512
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "anthropic"
_WEATHER_CASSETTE = _CASSETTES / "bedrock_tool_loop_cache_weather.json"
_FORECAST_CASSETTE = _CASSETTES / "bedrock_tool_loop_cache_forecast.json"
_ANSWER_CASSETTE = _CASSETTES / "bedrock_tool_loop_cache_answer.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {
    "input_rate": 1.10 / 1_000_000,
    "output_rate": 5.50 / 1_000_000,
    "cache_read_rate": 0.11 / 1_000_000,
    "cache_write_rate": 1.375 / 1_000_000,
}

_TASK = (
    "Answer using the tools, one step per turn. "
    "First call get_weather for Denver and for Seattle, both in the same turn. "
    "Then call get_forecast for whichever of the two is warmer. "
    "Then reply with one sentence."
)

_CITY_PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
_TOOLS = [
    Tool(
        function=FunctionDefinition(
            name="get_weather", description="Get the current weather for a city.", parameters=_CITY_PARAMS
        )
    ),
    Tool(
        function=FunctionDefinition(
            name="get_forecast", description="Get the three-day forecast for a city.", parameters=_CITY_PARAMS
        )
    ),
]

# A marker-only message: it contributes no content of its own, so the cache point lands on
# whatever precedes it — here, the merged tool-result turn.
_CACHE_MARKER = UserMessage(content=[CachePointContent()])


def _city(tool_call: Any) -> str:  # noqa: ANN401 — a ToolCall, kept loose to read the mapped arguments
    return str(json.loads(tool_call.function.arguments)["city"])


def _weather_result(city: str) -> str:
    """A fixed reading per city, so the model takes the same branch on every run."""
    return json.dumps({"city": city, "tempF": 78 if "denver" in city.lower() else 61})


@pytest.mark.verified
def test_cache_point_after_tool_results(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    cache_filler: Callable[..., str],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    question = UserMessage(content=[TextContent(text=f"{cache_filler(300)}\n\n{_TASK}"), CachePointContent()])
    ask: list[Message] = [question]

    def _weather(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
            _MODEL, ask, tools=_TOOLS, max_tokens=_MAX_TOKENS
        )

    weather = scenario(
        _WEATHER_CASSETTE, _weather, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth
    )
    assert weather.tool_calls is not None
    assert len(weather.tool_calls) > 1  # both cities in one turn, so the tool results merge into one turn
    assert {call.function.name for call in weather.tool_calls} == {"get_weather"}
    assert weather.usage is not None
    question_prefix = weather.usage.cache_creation_tokens
    assert question_prefix is not None
    assert question_prefix > 1024  # a real, substantial cache write (the question is >1024 tokens by design)
    assert weather.usage.cache_read_tokens is None  # nothing precedes the question in the cache
    assert_cost(weather, **_RATES)

    served: list[Message] = [
        *ask,
        weather.to_assistant_message(),
        *(ToolMessage(content=_weather_result(_city(call)), tool_call_id=call.id) for call in weather.tool_calls),
        _CACHE_MARKER,
    ]

    def _forecast(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
            _MODEL, served, tools=_TOOLS, max_tokens=_MAX_TOKENS
        )

    forecast = scenario(
        _FORECAST_CASSETTE, _forecast, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth
    )
    assert forecast.tool_calls is not None
    assert forecast.tool_calls[0].function.name == "get_forecast"
    assert forecast.usage is not None
    assert forecast.usage.cache_read_tokens == question_prefix  # the question prefix, read back whole
    tool_prefix = forecast.usage.cache_creation_tokens
    assert tool_prefix is not None
    assert tool_prefix > 0  # the merged tool-result turn extends the cached prefix
    assert_cost(forecast, **_RATES)

    forecast_call = forecast.tool_calls[0]
    resolved: list[Message] = [
        *served,
        forecast.to_assistant_message(),
        ToolMessage(
            content=json.dumps({"city": _city(forecast_call), "highsF": [79, 81, 77]}),
            tool_call_id=forecast_call.id,
        ),
        _CACHE_MARKER,
    ]

    def _answer(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
            _MODEL, resolved, tools=_TOOLS, max_tokens=_MAX_TOKENS
        )

    answer = scenario(_ANSWER_CASSETTE, _answer, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(answer, provider="anthropic-bedrock")
    assert answer.usage is not None
    # The whole point: the prefix read back grew by exactly the tool history the previous hop
    # wrote, so the tool results are inside the cached prefix and not merely behind it.
    assert answer.usage.cache_read_tokens == question_prefix + tool_prefix
    assert answer.usage.cache_creation_tokens is not None  # and this hop's tool result extends it again
    assert_cost(answer, **_RATES)
