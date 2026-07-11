"""Tests for Anthropic type mappers."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import UnsupportedFeatureError
from lmux.types import (
    AssistantMessage,
    CachePointContent,
    ChatChunk,
    ChatResponse,
    Cost,
    DeveloperMessage,
    FunctionCallDelta,
    FunctionCallResult,
    FunctionDefinition,
    ImageContent,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextContent,
    TextResponseFormat,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolChoiceFunction,
    ToolMessage,
    Usage,
    UserMessage,
)
from lmux_anthropic._mappers import (
    CostCalculator,
    map_content_block_delta,
    map_content_block_start,
    map_message_delta,
    map_message_response,
    map_message_start,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
    model_uses_adaptive_thinking,
)
from lmux_anthropic._wire import (
    WireContentBlockDeltaEvent,
    WireContentBlockStartEvent,
    WireMessage,
    WireMessageDeltaEvent,
    WireMessageStartEvent,
)

# MARK: Fixtures


@pytest.fixture
def cost_fn() -> CostCalculator:
    def _cost_fn(_model: str, _usage: Usage) -> Cost:
        return Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03)

    return _cost_fn


@pytest.fixture
def none_cost_fn() -> CostCalculator:
    def _cost_fn(_model: str, _usage: Usage) -> None:
        return None

    return _cost_fn


# MARK: map_messages


class TestMapMessages:
    def test_system_message_extracted(self) -> None:
        system, messages = map_messages([SystemMessage(content="Be helpful.")])
        assert system == "Be helpful."
        assert messages == []

    def test_developer_message_merged_into_system(self) -> None:
        system, messages = map_messages([DeveloperMessage(content="You are a dev assistant.")])
        assert system == "You are a dev assistant."
        assert messages == []

    def test_multiple_system_messages_concatenated(self) -> None:
        system, _ = map_messages(
            [
                SystemMessage(content="First."),
                DeveloperMessage(content="Second."),
                SystemMessage(content="Third."),
            ]
        )
        assert system == "First.\nSecond.\nThird."

    def test_user_message_text(self) -> None:
        system, messages = map_messages([UserMessage(content="Hello")])
        assert system is None
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_user_message_multimodal(self) -> None:
        _, messages = map_messages(
            [UserMessage(content=[TextContent(text="Look at this"), ImageContent(url="https://example.com/img.png")])]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                ],
            }
        ]

    def test_user_message_base64_image(self) -> None:
        _, messages = map_messages([UserMessage(content=[ImageContent(url="data:image/png;base64,iVBOR==")])])
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR=="},
        }

    def test_assistant_message_text(self) -> None:
        _, messages = map_messages([AssistantMessage(content="Hi there")])
        assert messages == [{"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}]

    def test_assistant_message_with_tool_calls(self) -> None:
        _, messages = map_messages(
            [
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'),
                        )
                    ]
                )
            ]
        )
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            }
        ]

    def test_assistant_message_text_and_tool_calls(self) -> None:
        _, messages = map_messages(
            [
                AssistantMessage(
                    content="Let me check.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'),
                        )
                    ],
                )
            ]
        )
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1]["type"] == "tool_use"

    def test_tool_message(self) -> None:
        _, messages = map_messages([ToolMessage(content="72°F", tool_call_id="call_1")])
        assert messages == [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "72°F"}]}
        ]

    def test_consecutive_tool_messages_merged(self) -> None:
        _, messages = map_messages(
            [
                ToolMessage(content="72°F", tool_call_id="call_1"),
                ToolMessage(content="Sunny", tool_call_id="call_2"),
            ]
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["tool_use_id"] == "call_1"
        assert content[1]["tool_use_id"] == "call_2"

    def test_tool_message_after_multimodal_user_not_merged(self) -> None:
        _, messages = map_messages(
            [
                UserMessage(content=[TextContent(text="Look"), ImageContent(url="https://example.com/img.png")]),
                ToolMessage(content="72°F", tool_call_id="call_1"),
            ]
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_result"

    def test_tool_message_after_user_not_merged(self) -> None:
        _, messages = map_messages(
            [
                UserMessage(content="Check this"),
                ToolMessage(content="72°F", tool_call_id="call_1"),
            ]
        )
        assert len(messages) == 2

    def test_mixed_messages(self) -> None:
        system, messages = map_messages(
            [
                SystemMessage(content="Be helpful."),
                UserMessage(content="Hi"),
                AssistantMessage(content="Hello!"),
            ]
        )
        assert system == "Be helpful."
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_no_system_returns_none(self) -> None:
        system, _ = map_messages([UserMessage(content="Hi")])
        assert system is None

    # MARK: map_tools

    def test_cache_point_attaches_to_preceding_block(self) -> None:
        _, messages = map_messages([UserMessage(content=[TextContent(text="Big context"), CachePointContent()])])
        assert messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Big context", "cache_control": {"type": "ephemeral"}}],
            }
        ]

    def test_cache_point_with_ttl(self) -> None:
        _, messages = map_messages(
            [UserMessage(content=[TextContent(text="Big context"), CachePointContent(ttl="1h")])]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Big context", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                ],
            }
        ]

    def test_cache_point_mid_message(self) -> None:
        _, messages = map_messages(
            [UserMessage(content=[TextContent(text="Stable"), CachePointContent(), TextContent(text="Varying")])]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Stable", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Varying"},
                ],
            }
        ]

    def test_marker_only_message_attaches_to_previous_string_message(self) -> None:
        _, messages = map_messages([UserMessage(content="Hello"), UserMessage(content=[CachePointContent()])])
        assert messages == [
            {"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}
        ]

    def test_marker_only_message_attaches_to_tool_result(self) -> None:
        _, messages = map_messages(
            [
                AssistantMessage(
                    content=None,
                    tool_calls=[ToolCall(id="t1", function=FunctionCallResult(name="f", arguments="{}"))],
                ),
                ToolMessage(content="result", tool_call_id="t1"),
                UserMessage(content=[CachePointContent()]),
            ]
        )
        assert messages[-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "result",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    def test_leading_cache_point_attaches_to_system(self) -> None:
        system, messages = map_messages(
            [SystemMessage(content="Be helpful."), UserMessage(content=[CachePointContent(), TextContent(text="Hi")])]
        )
        assert system == [{"type": "text", "text": "Be helpful.", "cache_control": {"type": "ephemeral"}}]
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_leading_cache_point_with_no_prefix_dropped(self) -> None:
        system, messages = map_messages([UserMessage(content=[CachePointContent(), TextContent(text="Hi")])])
        assert system is None
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_marker_only_first_message_dropped(self) -> None:
        system, messages = map_messages([UserMessage(content=[CachePointContent()])])
        assert system is None
        assert messages == []

    def test_attach_to_empty_content_message_is_noop(self) -> None:
        _, messages = map_messages([AssistantMessage(), UserMessage(content=[CachePointContent()])])
        assert messages == [{"role": "assistant", "content": []}]

    def test_system_cache_point_splits_at_marker_position(self) -> None:
        system, messages = map_messages(
            [
                SystemMessage(content="Stable A"),
                UserMessage(content=[CachePointContent()]),
                SystemMessage(content="Volatile B"),
            ]
        )
        # Only system text seen before the marker is cached; later text stays outside.
        assert system == [
            {"type": "text", "text": "Stable A", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Volatile B"},
        ]
        assert messages == []

    def test_leading_cache_point_before_any_system_is_dropped(self) -> None:
        system, messages = map_messages(
            [
                UserMessage(content=[CachePointContent(), TextContent(text="Q")]),
                SystemMessage(content="Late system"),
            ]
        )
        # Nothing preceded the marker, so system text arriving later is not cached.
        assert system == "Late system"
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]

    def test_first_system_cache_point_wins(self) -> None:
        system, _ = map_messages(
            [
                SystemMessage(content="A"),
                UserMessage(content=[CachePointContent(ttl="1h")]),
                UserMessage(content=[CachePointContent()]),
            ]
        )
        assert system == [{"type": "text", "text": "A", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]

    def test_first_cache_point_wins_within_message(self) -> None:
        _, messages = map_messages(
            [UserMessage(content=[TextContent(text="ctx"), CachePointContent(ttl="1h"), CachePointContent()])]
        )
        assert messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            }
        ]

    def test_marker_only_message_does_not_downgrade_existing_marker(self) -> None:
        _, messages = map_messages(
            [
                UserMessage(content=[TextContent(text="ctx"), CachePointContent(ttl="1h")]),
                UserMessage(content=[CachePointContent()]),
            ]
        )
        assert messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            }
        ]

    def test_first_leading_cache_point_wins_within_message(self) -> None:
        _, messages = map_messages(
            [
                UserMessage(content="prior"),
                UserMessage(content=[CachePointContent(ttl="1h"), CachePointContent(), TextContent(text="Q")]),
            ]
        )
        assert messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "prior", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        ]

    def test_marker_after_empty_string_message_is_dropped(self) -> None:
        _, messages = map_messages([UserMessage(content=""), UserMessage(content=[CachePointContent()])])
        # An empty string message has nothing cacheable; no empty text block is built.
        assert messages == [{"role": "user", "content": ""}]

    def test_originally_empty_content_list_is_forwarded(self) -> None:
        _, messages = map_messages([UserMessage(content=[])])
        assert messages == [{"role": "user", "content": []}]


class TestMapTools:
    def test_minimal_tool(self) -> None:
        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        result = map_tools(tools)
        assert result == [{"name": "get_weather", "input_schema": {"type": "object"}}]

    def test_tool_with_description_and_parameters(self) -> None:
        tools = [
            Tool(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get the weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            )
        ]
        result = map_tools(tools)
        assert result == [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]

    def test_none_parameters_defaults_to_object(self) -> None:
        tools = [Tool(function=FunctionDefinition(name="noop", parameters=None))]
        result = map_tools(tools)
        assert result[0]["input_schema"] == {"type": "object"}


# MARK: map_response_format


@pytest.fixture
def mock_add_additional_properties_false(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic._mappers.add_additional_properties_false")


class TestMapResponseFormat:
    def test_text_returns_none(self) -> None:
        assert map_response_format(TextResponseFormat()) is None

    def test_json_object_raises(self) -> None:
        with pytest.raises(UnsupportedFeatureError, match="JsonObjectResponseFormat is not supported"):
            _ = map_response_format(JsonObjectResponseFormat())

    def test_json_schema(self, mock_add_additional_properties_false: MagicMock) -> None:
        rf = JsonSchemaResponseFormat(
            name="person",
            json_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )
        result = map_response_format(rf)
        mock_add_additional_properties_false.assert_called_once()
        assert result == {
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "additionalProperties": False, "properties": {}},
            }
        }

    def test_json_schema_patches_nested_objects(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="test",
            json_schema={
                "type": "object",
                "properties": {
                    "inner": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
            },
        )
        result = map_response_format(rf)
        assert result is not None
        fmt = cast("dict[str, Any]", cast("object", result["format"]))
        schema = cast("dict[str, Any]", fmt["schema"])
        assert schema["additionalProperties"] is False
        assert schema["properties"]["inner"]["additionalProperties"] is False

    def test_json_schema_patches_objects_inside_arrays(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="test",
            json_schema={
                "type": "object",
                "anyOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "object", "properties": {"b": {"type": "string"}}},
                ],
                "required": ["a"],  # list of non-dict items to cover the skip branch
            },
        )
        result = map_response_format(rf)
        assert result is not None
        fmt = cast("dict[str, Any]", cast("object", result["format"]))
        schema = cast("dict[str, Any]", fmt["schema"])
        assert schema["additionalProperties"] is False
        assert schema["anyOf"][0]["additionalProperties"] is False
        assert schema["anyOf"][1]["additionalProperties"] is False

    def test_json_schema_preserves_existing_additional_properties(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="test",
            json_schema={"type": "object", "additionalProperties": True},
        )
        result = map_response_format(rf)
        assert result is not None
        fmt = cast("dict[str, Any]", cast("object", result["format"]))
        schema = cast("dict[str, Any]", fmt["schema"])
        assert schema["additionalProperties"] is True


# MARK: map_message_response


class TestMapMessageResponse:
    def _usage(self, **overrides: object) -> dict[str, Any]:
        usage: dict[str, Any] = {"input_tokens": 10, "output_tokens": 5}
        usage.update(overrides)
        return usage

    def test_text_response(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03),
            model="claude-sonnet-4-6",
            provider="anthropic",
            finish_reason="stop",
        )

    def test_tool_use_response(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "NYC"}}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.content is None
        assert result.tool_calls == [
            ToolCall(id="call_1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        ]
        assert result.finish_reason == "tool_calls"

    def test_thinking_blocks_extracted(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "Answer"},
            ],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.content == "Answer"
        assert result.reasoning == "Let me think about this..."

    def test_thinking_blocks_only_no_text(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "Deep thought..."}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.content is None
        assert result.reasoning == "Deep thought..."

    def test_multiple_text_blocks_joined(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.content == "Part 1\nPart 2"

    def test_thinking_with_tool_use_and_text(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "Reasoning..."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "NYC"}},
                {"type": "tool_use", "id": "call_2", "name": "get_time", "input": {}},
                {"type": "text", "text": "Here's the weather"},
            ],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.reasoning == "Reasoning..."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.content == "Here's the weather"

    def test_unknown_block_type_ignored(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "some_future_block"}, {"type": "text", "text": "Answer"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.content == "Answer"
        assert result.reasoning is None

    def test_none_cost_when_unknown_model(self, none_cost_fn: CostCalculator) -> None:
        message = {
            "model": "unknown-model",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", none_cost_fn)
        assert result.cost is None

    def test_cache_tokens_mapped(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(
                input_tokens=100, output_tokens=50, cache_read_input_tokens=20, cache_creation_input_tokens=10
            ),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.usage is not None
        # input_tokens is normalized to the inclusive total (100 + 20 + 10)
        assert result.usage.input_tokens == 130
        assert result.usage.cache_read_tokens == 20
        assert result.usage.cache_creation_tokens == 10

    def test_none_stop_reason(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": None,
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.finish_reason is None

    def test_missing_stop_reason(self, cost_fn: CostCalculator) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", cost_fn)
        assert result.finish_reason is None

    def test_breakdown_mapped_from_cache_creation(self) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(
                input_tokens=100,
                cache_creation_input_tokens=2000,
                cache_creation={"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 1500},
            ),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", lambda _m, _u: None)
        assert result.usage is not None
        assert result.usage.input_tokens == 2100
        assert result.usage.cache_creation_tokens_by_ttl == {"5m": 500, "1h": 1500}

    def test_zero_breakdown_fields_omitted(self) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(
                cache_creation_input_tokens=1500,
                cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1500},
            ),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", lambda _m, _u: None)
        assert result.usage is not None
        assert result.usage.cache_creation_tokens_by_ttl == {"1h": 1500}

    def test_all_zero_breakdown_is_none(self) -> None:
        message = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": self._usage(cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}),
        }
        result = map_message_response(WireMessage.model_validate(message), "anthropic", lambda _m, _u: None)
        assert result.usage is not None
        assert result.usage.cache_creation_tokens_by_ttl is None


class TestMapMessageStart:
    def test_extracts_model_and_usage(self) -> None:
        event = {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-4-6",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 5,
                },
            },
        }
        model, usage = map_message_start(WireMessageStartEvent.model_validate(event))
        assert model == "claude-sonnet-4-6"
        assert usage == Usage(input_tokens=65, output_tokens=0, cache_read_tokens=10, cache_creation_tokens=5)


class TestMapContentBlockStart:
    def test_tool_use_block(self) -> None:
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}},
        }
        result = map_content_block_start(WireContentBlockStartEvent.model_validate(event))
        assert result == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, id="call_1", type="function", function=FunctionCallDelta(name="get_weather"))
            ]
        )

    def test_text_block_returns_none(self) -> None:
        event = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        assert map_content_block_start(WireContentBlockStartEvent.model_validate(event)) is None


class TestMapContentBlockDelta:
    def test_text_delta(self) -> None:
        event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
        result = map_content_block_delta(WireContentBlockDeltaEvent.model_validate(event))
        assert result == ChatChunk(delta="Hello")

    def test_input_json_delta(self) -> None:
        event = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
        result = map_content_block_delta(WireContentBlockDeltaEvent.model_validate(event))
        assert result == ChatChunk(
            tool_call_deltas=[ToolCallDelta(index=1, function=FunctionCallDelta(arguments='{"city":'))]
        )

    def test_thinking_delta_returns_reasoning(self) -> None:
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
        }
        result = map_content_block_delta(WireContentBlockDeltaEvent.model_validate(event))
        assert result is not None
        assert result.reasoning_delta == "Let me think..."

    def test_unknown_delta_type_returns_none(self) -> None:
        event = {"type": "content_block_delta", "index": 0, "delta": {"type": "some_future_delta"}}
        assert map_content_block_delta(WireContentBlockDeltaEvent.model_validate(event)) is None


class TestMapMessageDelta:
    def test_final_chunk_with_usage(self) -> None:
        event = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 50}}
        start_usage = Usage(input_tokens=100, output_tokens=0, cache_read_tokens=10, cache_creation_tokens=5)
        result = map_message_delta(WireMessageDeltaEvent.model_validate(event), start_usage)
        assert result == ChatChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_creation_tokens=5),
        )

    def test_delta_usage_carries_breakdown(self) -> None:
        start_usage = Usage(
            input_tokens=2100,
            output_tokens=0,
            cache_creation_tokens=2000,
            cache_creation_tokens_by_ttl={"1h": 2000},
        )
        event = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}
        chunk = map_message_delta(WireMessageDeltaEvent.model_validate(event), start_usage)
        assert chunk.usage == Usage(
            input_tokens=2100,
            output_tokens=42,
            cache_creation_tokens=2000,
            cache_creation_tokens_by_ttl={"1h": 2000},
        )


class TestMapToolChoice:
    def test_auto(self) -> None:
        assert map_tool_choice("auto") == {"type": "auto"}

    def test_required(self) -> None:
        assert map_tool_choice("required") == {"type": "any"}

    def test_none(self) -> None:
        assert map_tool_choice("none") == {"type": "none"}

    def test_specific_function(self) -> None:
        assert map_tool_choice(ToolChoiceFunction(name="get_weather")) == {"type": "tool", "name": "get_weather"}


class TestModelUsesAdaptiveThinking:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
            "global.anthropic.claude-opus-4-8",
            "global.anthropic.claude-sonnet-5",
            "us.anthropic.claude-opus-4-6-v1",
            "claude-opus-4-6@20260201",
            "claude-sonnet-5@20260101",
        ],
    )
    def test_adaptive_generations(self, model: str) -> None:
        assert model_uses_adaptive_thinking(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-1",
            "claude-opus-4-5",
            "claude-opus-4",
            "claude-sonnet-4-5",
            "claude-sonnet-4",
            "claude-haiku-4-5",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1",
            "claude-opus-4-5@20251101",
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "not-a-model",
            "",
        ],
    )
    def test_legacy_and_unparseable_generations(self, model: str) -> None:
        assert model_uses_adaptive_thinking(model) is False
