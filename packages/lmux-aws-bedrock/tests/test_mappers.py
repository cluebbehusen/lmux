"""Tests for AWS Bedrock Converse API type mappers."""

import base64
import json
from typing import Any

import pytest

from lmux.exceptions import UnsupportedFeatureError
from lmux.types import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    ContentPart,
    Cost,
    DeveloperMessage,
    EmbeddingResponse,
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
from lmux_aws_bedrock._mappers import (
    build_embedding_request_body,
    map_converse_response,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_stream_event,
    map_tools,
)

# MARK: Fixtures


@pytest.fixture
def noop_cost_fn() -> Any:  # noqa: ANN401
    def _fn(model: str, usage: Usage) -> Cost:
        return Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0)

    return _fn


@pytest.fixture
def none_cost_fn() -> Any:  # noqa: ANN401
    def _fn(model: str, usage: Usage) -> None:
        return None

    return _fn


# MARK: map_messages


class TestMapMessages:
    def test_system_message_extracted(self) -> None:
        system, messages = map_messages([SystemMessage(content="Be helpful.")])
        assert system == [{"text": "Be helpful."}]
        assert messages == []

    def test_developer_message_extracted(self) -> None:
        system, messages = map_messages([DeveloperMessage(content="Be concise.")])
        assert system == [{"text": "Be concise."}]
        assert messages == []

    def test_user_message_text(self) -> None:
        system, messages = map_messages([UserMessage(content="Hello")])
        assert system is None
        assert messages == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_user_message_multimodal_base64(self) -> None:
        raw_bytes = b"\x89PNG\r\n"
        b64_data = base64.b64encode(raw_bytes).decode()
        data_uri = f"data:image/png;base64,{b64_data}"
        parts = [TextContent(text="What is this?"), ImageContent(url=data_uri)]
        system, messages = map_messages([UserMessage(content=parts)])

        assert system is None
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[0] == {"text": "What is this?"}
        assert content[1] == {"image": {"format": "png", "source": {"bytes": raw_bytes}}}

    def test_user_message_multimodal_url_raises(self) -> None:
        parts: list[ContentPart] = [ImageContent(url="https://example.com/image.png")]
        with pytest.raises(UnsupportedFeatureError, match="base64 data URIs"):
            map_messages([UserMessage(content=parts)])

    def test_assistant_message_text(self) -> None:
        system, messages = map_messages([AssistantMessage(content="Hi!")])
        assert system is None
        assert messages == [{"role": "assistant", "content": [{"text": "Hi!"}]}]

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        system, messages = map_messages([AssistantMessage(content="Let me check.", tool_calls=[tc])])

        assert system is None
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [
            {"text": "Let me check."},
            {
                "toolUse": {
                    "toolUseId": "tc1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            },
        ]

    def test_assistant_message_tool_calls_no_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        system, messages = map_messages([AssistantMessage(content=None, tool_calls=[tc])])

        assert system is None
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [
            {"toolUse": {"toolUseId": "tc1", "name": "f", "input": {}}},
        ]

    def test_tool_message(self) -> None:
        system, messages = map_messages([ToolMessage(content="72F", tool_call_id="tc1")])
        assert system is None
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tc1",
                            "content": [{"text": "72F"}],
                            "status": "success",
                        }
                    }
                ],
            }
        ]

    def test_consecutive_tool_messages_merged(self) -> None:
        system, messages = map_messages(
            [
                ToolMessage(content="result1", tool_call_id="tc1"),
                ToolMessage(content="result2", tool_call_id="tc2"),
            ]
        )
        assert system is None
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["toolResult"]["toolUseId"] == "tc1"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert content[1]["toolResult"]["toolUseId"] == "tc2"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    def test_tool_message_after_user_not_merged(self) -> None:
        system, messages = map_messages(
            [
                UserMessage(content="hello"),
                ToolMessage(content="result", tool_call_id="tc1"),
            ]
        )
        assert system is None
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [{"text": "hello"}]
        assert messages[1]["role"] == "user"
        assert "toolResult" in messages[1]["content"][0]

    def test_mixed_messages(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        system, messages = map_messages(msgs)
        assert system == [{"text": "sys"}]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_no_system_returns_none(self) -> None:
        system, _messages = map_messages([UserMessage(content="hi")])
        assert system is None


# MARK: map_tools


class TestMapTools:
    def test_full_tool(self) -> None:
        tools = [
            Tool(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            )
        ]
        result = map_tools(tools)
        assert result == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}},
                    }
                }
            ]
        }

    def test_minimal_tool(self) -> None:
        tools = [Tool(function=FunctionDefinition(name="noop"))]
        result = map_tools(tools)
        assert result == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "noop",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }


# MARK: map_response_format


class TestMapResponseFormat:
    def test_text_returns_none(self) -> None:
        assert map_response_format(TextResponseFormat()) is None

    def test_json_object_returns_output_config(self) -> None:
        result = map_response_format(JsonObjectResponseFormat())
        assert result == {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": '{"type": "object"}',
                        "name": "json_object",
                    }
                },
            }
        }

    def test_json_schema_returns_output_config(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="test",
            description="Test schema",
            json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        result = map_response_format(rf)
        assert result == {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": '{"properties": {"city": {"type": "string"}}, "type": "object"}',
                        "name": "test",
                        "description": "Test schema",
                    }
                },
            }
        }

    def test_json_schema_without_description_returns_output_config(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        result = map_response_format(rf)
        assert result == {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": '{"type": "object"}',
                        "name": "test",
                    }
                },
            }
        }


# MARK: map_converse_response


class TestMapConverseResponse:
    def test_text_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="anthropic.claude-3",
            provider="aws-bedrock",
            finish_reason="stop",
        )

    def test_tool_use_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tc1",
                                "name": "get_weather",
                                "input": {"city": "NYC"},
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 15, "outputTokens": 8},
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result == ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'),
                )
            ],
            usage=Usage(input_tokens=15, output_tokens=8),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="anthropic.claude-3",
            provider="aws-bedrock",
            finish_reason="tool_calls",
        )

    def test_without_usage(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None

    def test_with_cache_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"text": "cached"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheReadInputTokenCount": 50,
                "cacheWriteInputTokenCount": 20,
            },
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50
        assert result.usage.cache_creation_tokens == 20

    def test_empty_content_blocks(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 0},
        }
        result = map_converse_response(response, "m", "aws-bedrock", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None

    def test_unknown_content_block_skipped(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"guardContent": {"text": "blocked"}}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 0},
        }
        result = map_converse_response(response, "m", "aws-bedrock", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None

    def test_reasoning_content_extracted(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "Let me think..."}}},
                        {"text": "Answer"},
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning == "Let me think..."

    def test_reasoning_content_empty_text(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": ""}}},
                        {"text": "Answer"},
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = map_converse_response(response, "anthropic.claude-3", "aws-bedrock", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning is None

    def test_cost_none_for_unknown_model(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = map_converse_response(response, "unknown-model", "aws-bedrock", none_cost_fn)
        assert result.cost is None


# MARK: _map_stop_reason (tested via map_converse_response)


class TestMapStopReason:
    def _make_response(self, stop_reason: str | None) -> Any:  # noqa: ANN401
        response: Any = {
            "output": {"message": {"content": [{"text": "x"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        if stop_reason is not None:
            response["stopReason"] = stop_reason
        return response

    def test_end_turn(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_converse_response(self._make_response("end_turn"), "m", "p", noop_cost_fn)
        assert result.finish_reason == "stop"

    def test_tool_use(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_converse_response(self._make_response("tool_use"), "m", "p", noop_cost_fn)
        assert result.finish_reason == "tool_calls"

    def test_max_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_converse_response(self._make_response("max_tokens"), "m", "p", noop_cost_fn)
        assert result.finish_reason == "length"

    def test_none(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_converse_response(self._make_response(None), "m", "p", noop_cost_fn)
        assert result.finish_reason is None

    def test_unknown_passthrough(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_converse_response(self._make_response("something_new"), "m", "p", noop_cost_fn)
        assert result.finish_reason == "something_new"


# MARK: map_stream_event


class TestMapStreamEvent:
    def test_content_block_delta_text(self) -> None:
        event: Any = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Hello"},
            }
        }
        result = map_stream_event(event)
        assert result == ChatChunk(delta="Hello")

    def test_content_block_delta_tool_use(self) -> None:
        event: Any = {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"city":'}},
            }
        }
        result = map_stream_event(event)
        assert result == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=1,
                    function=FunctionCallDelta(arguments='{"city":'),
                )
            ]
        )

    def test_content_block_start_tool_use(self) -> None:
        event: Any = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": "tc1",
                        "name": "get_weather",
                    }
                },
            }
        }
        result = map_stream_event(event)
        assert result == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=0,
                    id="tc1",
                    type="function",
                    function=FunctionCallDelta(name="get_weather"),
                )
            ]
        )

    def test_message_stop(self) -> None:
        event: Any = {"messageStop": {"stopReason": "end_turn"}}
        result = map_stream_event(event)
        assert result == ChatChunk(finish_reason="stop")

    def test_metadata_with_usage(self) -> None:
        event: Any = {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }
        }
        result = map_stream_event(event)
        assert result == ChatChunk(usage=Usage(input_tokens=10, output_tokens=5))

    def test_metadata_without_usage(self) -> None:
        event: Any = {"metadata": {"requestId": "abc"}}
        result = map_stream_event(event)
        assert result is None

    def test_unknown_event_returns_none(self) -> None:
        event: Any = {"messageStart": {"role": "assistant"}}
        result = map_stream_event(event)
        assert result is None

    def test_content_block_start_text_returns_none(self) -> None:
        event: Any = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"text": ""},
            }
        }
        result = map_stream_event(event)
        assert result is None

    def test_content_block_delta_reasoning(self) -> None:
        event: Any = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "thinking..."}},
            }
        }
        result = map_stream_event(event)
        assert result is not None
        assert result.reasoning_delta == "thinking..."

    def test_content_block_delta_reasoning_no_text(self) -> None:
        event: Any = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {}},
            }
        }
        result = map_stream_event(event)
        assert result is None

    def test_content_block_delta_unknown_returns_none(self) -> None:
        event: Any = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"unknownType": "data"},
            }
        }
        result = map_stream_event(event)
        assert result is None


# MARK: map_embedding_response


class TestMapEmbeddingResponse:
    def test_basic(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response_body: dict[str, Any] = {
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 5,
        }
        result = map_embedding_response(response_body, "amazon.titan-embed-text-v1", "aws-bedrock", noop_cost_fn)
        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="amazon.titan-embed-text-v1",
            provider="aws-bedrock",
        )

    def test_cost_calculation(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response_body: dict[str, Any] = {
            "embedding": [0.5],
            "inputTextTokenCount": 3,
        }
        result = map_embedding_response(response_body, "unknown-model", "aws-bedrock", none_cost_fn)
        assert result.cost is None
        assert result.usage == Usage(input_tokens=3, output_tokens=0)


# MARK: build_embedding_request_body


class TestBuildEmbeddingRequestBody:
    def test_produces_valid_json(self) -> None:
        body = build_embedding_request_body("Hello, world!")
        parsed = json.loads(body)
        assert parsed == {"inputText": "Hello, world!"}

    def test_with_dimensions(self) -> None:
        body = build_embedding_request_body("Hello!", dimensions=256)
        parsed = json.loads(body)
        assert parsed == {"inputText": "Hello!", "dimensions": 256}
