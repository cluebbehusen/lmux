"""Tests for Anthropic type mappers."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import UnsupportedFeatureError
from lmux.types import (
    AssistantMessage,
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
    map_tools,
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
        assert content[1]["type"] == "tool_use"  # pyright: ignore[reportIndexIssue]

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
        assert content[0]["tool_use_id"] == "call_1"  # pyright: ignore[reportIndexIssue, reportGeneralTypeIssues]
        assert content[1]["tool_use_id"] == "call_2"  # pyright: ignore[reportIndexIssue, reportGeneralTypeIssues]

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
        assert content[0]["type"] == "tool_result"  # pyright: ignore[reportIndexIssue]

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
        fmt = cast("dict[str, Any]", cast("object", result["format"]))  # pyright: ignore[reportTypedDictNotRequiredAccess]
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
        fmt = cast("dict[str, Any]", cast("object", result["format"]))  # pyright: ignore[reportTypedDictNotRequiredAccess]
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
        fmt = cast("dict[str, Any]", cast("object", result["format"]))  # pyright: ignore[reportTypedDictNotRequiredAccess]
        schema = cast("dict[str, Any]", fmt["schema"])
        assert schema["additionalProperties"] is True


# MARK: map_message_response


class TestMapMessageResponse:
    def test_text_response(self, cost_fn: CostCalculator) -> None:
        message = MagicMock()
        message.content = [MagicMock(type="text", text="Hello!")]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
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
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "NYC"}

        message = MagicMock()
        message.content = [tool_block]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "tool_use"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.content is None
        assert result.tool_calls == [
            ToolCall(id="call_1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        ]
        assert result.finish_reason == "tool_calls"

    def test_thinking_blocks_extracted(self, cost_fn: CostCalculator) -> None:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me think about this..."
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Answer"

        message = MagicMock()
        message.content = [thinking_block, text_block]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.content == "Answer"
        assert result.reasoning == "Let me think about this..."

    def test_thinking_blocks_only_no_text(self, cost_fn: CostCalculator) -> None:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Deep thought..."

        message = MagicMock()
        message.content = [thinking_block]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.content is None
        assert result.reasoning == "Deep thought..."

    def test_multiple_text_blocks_joined(self, cost_fn: CostCalculator) -> None:
        message = MagicMock()
        message.content = [
            MagicMock(type="text", text="Part 1"),
            MagicMock(type="text", text="Part 2"),
        ]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.content == "Part 1\nPart 2"

    def test_thinking_with_tool_use_and_text(self, cost_fn: CostCalculator) -> None:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Reasoning..."

        tool_block1 = MagicMock()
        tool_block1.type = "tool_use"
        tool_block1.id = "call_1"
        tool_block1.name = "get_weather"
        tool_block1.input = {"city": "NYC"}

        tool_block2 = MagicMock()
        tool_block2.type = "tool_use"
        tool_block2.id = "call_2"
        tool_block2.name = "get_time"
        tool_block2.input = {}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here's the weather"

        message = MagicMock()
        message.content = [thinking_block, tool_block1, tool_block2, text_block]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "tool_use"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.reasoning == "Reasoning..."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.content == "Here's the weather"

    def test_unknown_block_type_ignored(self, cost_fn: CostCalculator) -> None:
        unknown_block = MagicMock()
        unknown_block.type = "some_future_block"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Answer"

        message = MagicMock()
        message.content = [unknown_block, text_block]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.content == "Answer"
        assert result.reasoning is None

    def test_none_cost_when_unknown_model(self, none_cost_fn: CostCalculator) -> None:
        message = MagicMock()
        message.content = [MagicMock(type="text", text="Hello")]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "unknown-model"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", none_cost_fn)
        assert result.cost is None

    def test_cache_tokens_mapped(self, cost_fn: CostCalculator) -> None:
        message = MagicMock()
        message.content = [MagicMock(type="text", text="Hello")]
        message.usage = MagicMock(
            input_tokens=100, output_tokens=50, cache_read_input_tokens=20, cache_creation_input_tokens=10
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = "end_turn"

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 20
        assert result.usage.cache_creation_tokens == 10

    def test_none_stop_reason(self, cost_fn: CostCalculator) -> None:
        message = MagicMock()
        message.content = [MagicMock(type="text", text="Hello")]
        message.usage = MagicMock(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )
        message.model = "claude-sonnet-4-6"
        message.stop_reason = None

        result = map_message_response(message, "anthropic", cost_fn)
        assert result.finish_reason is None


# MARK: Streaming mappers


class TestMapMessageStart:
    def test_extracts_usage(self) -> None:
        event = MagicMock()
        event.message.usage = MagicMock(
            input_tokens=50, output_tokens=0, cache_read_input_tokens=10, cache_creation_input_tokens=5
        )
        result = map_message_start(event)
        assert result == Usage(input_tokens=50, output_tokens=0, cache_read_tokens=10, cache_creation_tokens=5)


class TestMapContentBlockStart:
    def test_tool_use_block(self) -> None:
        event = MagicMock()
        event.content_block.type = "tool_use"
        event.content_block.id = "call_1"
        event.content_block.name = "get_weather"
        event.index = 0

        result = map_content_block_start(event)
        assert result == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, id="call_1", type="function", function=FunctionCallDelta(name="get_weather"))
            ]
        )

    def test_text_block_returns_none(self) -> None:
        event = MagicMock()
        event.content_block.type = "text"
        assert map_content_block_start(event) is None


class TestMapContentBlockDelta:
    def test_text_delta(self) -> None:
        event = MagicMock()
        event.delta.type = "text_delta"
        event.delta.text = "Hello"
        event.index = 0

        result = map_content_block_delta(event)
        assert result == ChatChunk(delta="Hello")

    def test_input_json_delta(self) -> None:
        event = MagicMock()
        event.delta.type = "input_json_delta"
        event.delta.partial_json = '{"city":'
        event.index = 1

        result = map_content_block_delta(event)
        assert result == ChatChunk(
            tool_call_deltas=[ToolCallDelta(index=1, function=FunctionCallDelta(arguments='{"city":'))]
        )

    def test_thinking_delta_returns_reasoning(self) -> None:
        event = MagicMock()
        event.delta.type = "thinking_delta"
        event.delta.thinking = "Let me think..."
        result = map_content_block_delta(event)
        assert result is not None
        assert result.reasoning_delta == "Let me think..."

    def test_unknown_delta_type_returns_none(self) -> None:
        event = MagicMock()
        event.delta.type = "some_future_delta"
        assert map_content_block_delta(event) is None


class TestMapMessageDelta:
    def test_final_chunk_with_usage(self) -> None:
        event = MagicMock()
        event.delta.stop_reason = "end_turn"
        event.usage.output_tokens = 50

        start_usage = Usage(input_tokens=100, output_tokens=0, cache_read_tokens=10, cache_creation_tokens=5)
        result = map_message_delta(event, start_usage)

        assert result == ChatChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_creation_tokens=5),
        )
