"""Tests for OpenAI JSON mappers."""

from typing import Any

import pytest

from lmux.types import (
    AssistantMessage,
    CachePointContent,
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
    ResponseInputFunctionCallOutput,
    ResponseInputMessage,
    ResponseResponse,
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
from lmux_openai._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_response_input,
    map_responses_response,
    map_tool_choice,
    map_tools,
)
from lmux_openai._wire import (
    WireChunk,
    WireCompletion,
    WireEmbeddingResponse,
    WireResponsesResponse,
)

# MARK: Fixtures


@pytest.fixture
def noop_cost_fn() -> Any:  # noqa: ANN401
    def _fn(_model: str, _usage: Usage) -> Cost:
        return Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0)

    return _fn


@pytest.fixture
def none_cost_fn() -> Any:  # noqa: ANN401
    def _fn(_model: str, _usage: Usage) -> None:
        return None

    return _fn


@pytest.fixture
def chat_completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# MARK: map_messages


class TestMapMessages:
    def test_system_message(self) -> None:
        assert map_messages([SystemMessage(content="Be helpful.")]) == [{"role": "system", "content": "Be helpful."}]

    def test_developer_message(self) -> None:
        assert map_messages([DeveloperMessage(content="Be concise.")]) == [
            {"role": "developer", "content": "Be concise."}
        ]

    def test_user_message_text(self) -> None:
        assert map_messages([UserMessage(content="Hello")]) == [{"role": "user", "content": "Hello"}]

    def test_originally_empty_content_list_is_forwarded(self) -> None:
        assert map_messages([UserMessage(content=[])]) == [{"role": "user", "content": []}]

    def test_user_message_multimodal(self) -> None:
        parts: list[ContentPart] = [TextContent(text="What?"), ImageContent(url="https://img.png", detail="high")]
        result = map_messages([UserMessage(content=parts)])
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What?"},
                    {"type": "image_url", "image_url": {"url": "https://img.png", "detail": "high"}},
                ],
            }
        ]

    def test_assistant_message_text(self) -> None:
        assert map_messages([AssistantMessage(content="Hi!")]) == [{"role": "assistant", "content": "Hi!"}]

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        result = map_messages([AssistantMessage(tool_calls=[tc])])
        assert result == [
            {
                "role": "assistant",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            }
        ]

    def test_tool_message(self) -> None:
        assert map_messages([ToolMessage(content="result", tool_call_id="tc1")]) == [
            {"role": "tool", "content": "result", "tool_call_id": "tc1"}
        ]

    def test_cache_points_dropped(self) -> None:
        result = map_messages([UserMessage(content=[TextContent(text="Hi"), CachePointContent()])])
        assert result == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_marker_only_message_skipped(self) -> None:
        result = map_messages([UserMessage(content="Hello"), UserMessage(content=[CachePointContent()])])
        assert result == [{"role": "user", "content": "Hello"}]


# MARK: map_tools


class TestMapTools:
    def test_minimal_tool(self) -> None:
        assert map_tools([Tool(function=FunctionDefinition(name="f"))]) == [
            {"type": "function", "function": {"name": "f"}}
        ]

    def test_full_tool(self) -> None:
        tools = [
            Tool(
                function=FunctionDefinition(
                    name="get_weather", description="Get weather", parameters={"type": "object"}, strict=True
                )
            )
        ]
        assert map_tools(tools) == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ]


# MARK: map_tool_choice


class TestMapToolChoice:
    def test_auto(self) -> None:
        assert map_tool_choice("auto") == "auto"

    def test_required(self) -> None:
        assert map_tool_choice("required") == "required"

    def test_none(self) -> None:
        assert map_tool_choice("none") == "none"

    def test_specific_function(self) -> None:
        assert map_tool_choice(ToolChoiceFunction(name="get_weather")) == {
            "type": "function",
            "function": {"name": "get_weather"},
        }


# MARK: map_response_format


class TestMapResponseFormat:
    def test_text(self) -> None:
        assert map_response_format(TextResponseFormat()) == {"type": "text"}

    def test_json_object(self) -> None:
        assert map_response_format(JsonObjectResponseFormat()) == {"type": "json_object"}

    def test_json_schema_minimal(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        assert map_response_format(rf) == {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }

    def test_json_schema_full(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"}, description="A test", strict=True)
        assert map_response_format(rf) == {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}, "description": "A test", "strict": True},
        }


# MARK: map_response_input


class TestMapResponseInput:
    def test_string_passthrough(self) -> None:
        assert map_response_input("Hello") == "Hello"

    def test_list_of_items(self) -> None:
        items = [
            ResponseInputMessage(role="user", content="call the tool"),
            ResponseInputFunctionCallOutput(call_id="call_1", output='{"result": 42}'),
        ]
        assert map_response_input(items) == [
            {"role": "user", "content": "call the tool"},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 42}'},
        ]


# MARK: map_chat_completion


class TestMapChatCompletion:
    def test_basic(self, chat_completion: dict[str, Any], noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_chat_completion(WireCompletion.model_validate(chat_completion), "openai", noop_cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-4o",
            provider="openai",
            finish_reason="stop",
        )

    def test_with_tool_calls_filters_non_function(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                            {"id": "tc2", "type": "custom", "custom": {}},
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "openai", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls == [ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments='{"x": 1}'))]

    def test_reasoning_content(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = {
            "id": "c",
            "model": "o3",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi", "reasoning_content": "thinking"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "openai", noop_cost_fn)
        assert result.reasoning == "thinking"

    def test_with_cache_read_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = {
            "id": "c",
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 50}},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_with_cache_write_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = {
            "id": "c",
            "model": "gpt-5.6-sol",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 6},
            },
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 2
        assert result.usage.cache_creation_tokens == 6

    def test_with_reasoning_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = {
            "id": "c",
            "model": "o3",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.reasoning_tokens == 20

    def test_none_usage(self, chat_completion: dict[str, Any], noop_cost_fn: Any) -> None:  # noqa: ANN401
        del chat_completion["usage"]
        result = map_chat_completion(WireCompletion.model_validate(chat_completion), "openai", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None

    def test_cost_none_when_unknown(self, chat_completion: dict[str, Any], none_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_chat_completion(WireCompletion.model_validate(chat_completion), "openai", none_cost_fn)
        assert result.cost is None


# MARK: map_chat_chunk


class TestMapChatChunk:
    def test_content_delta(self) -> None:
        chunk = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hello"}}]}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "openai") == ChatChunk(
            delta="Hello", model="gpt-4o", provider="openai"
        )

    def test_reasoning_delta(self) -> None:
        chunk = {
            "model": "o3",
            "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": "hmm"}}],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "openai").reasoning_delta == "hmm"

    def test_tool_call_delta(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"x":'},
                            }
                        ]
                    },
                }
            ],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "openai").tool_call_deltas == [
            ToolCallDelta(index=0, id="tc1", type="function", function=FunctionCallDelta(name="f", arguments='{"x":'))
        ]

    def test_tool_call_delta_without_function(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{"index": 0, "id": "tc1"}]}}],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "openai").tool_call_deltas == [
            ToolCallDelta(index=0, id="tc1", type=None, function=None)
        ]

    def test_final_chunk_with_usage(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "openai") == ChatChunk(
            finish_reason="stop", usage=Usage(input_tokens=10, output_tokens=5), model="gpt-4o", provider="openai"
        )

    def test_usage_chunk_with_cache_write(self) -> None:
        chunk = {
            "model": "gpt-5.6-sol",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1, "cache_write_tokens": 4},
            },
        }
        result = map_chat_chunk(WireChunk.model_validate(chunk), "openai")
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 1
        assert result.usage.cache_creation_tokens == 4

    def test_empty_choices(self) -> None:
        assert map_chat_chunk(WireChunk.model_validate({"model": "gpt-4o", "choices": []}), "openai") == ChatChunk(
            model="gpt-4o", provider="openai"
        )

    def test_missing_choices(self) -> None:
        assert map_chat_chunk(WireChunk.model_validate({"model": "gpt-4o"}), "openai") == ChatChunk(
            model="gpt-4o", provider="openai"
        )


# MARK: map_embedding_response


class TestMapEmbeddingResponse:
    def test_single_embedding(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        assert map_embedding_response(
            WireEmbeddingResponse.model_validate(response), "openai", noop_cost_fn
        ) == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="text-embedding-3-small",
            provider="openai",
        )

    def test_multiple_embeddings_sorted_by_index(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        assert map_embedding_response(
            WireEmbeddingResponse.model_validate(response), "openai", noop_cost_fn
        ).embeddings == [[0.1, 0.2], [0.3, 0.4]]


# MARK: map_responses_response


class TestMapResponsesResponse:
    def test_basic(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "id": "resp_123",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        assert map_responses_response(
            WireResponsesResponse.model_validate(response), "openai", noop_cost_fn
        ) == ResponseResponse(
            id="resp_123",
            output_text="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-4o",
            provider="openai",
        )

    def test_output_text_skips_non_message_and_non_text(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "id": "resp_1",
            "model": "gpt-4o",
            "output": [
                {"type": "reasoning", "content": [{"type": "output_text", "text": "ignored"}]},
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "no"},
                        {"type": "output_text", "text": "part-a "},
                        {"type": "output_text", "text": "part-b"},
                    ],
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        assert (
            map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn).output_text
            == "part-a part-b"
        )

    def test_output_missing(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"id": "r", "model": "gpt-4o", "usage": {"input_tokens": 1, "output_tokens": 1}}
        assert (
            map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn).output_text
            == ""
        )

    def test_with_cached_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "id": "r",
            "model": "gpt-4o",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 50}},
        }
        result = map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_with_cache_write_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "id": "r",
            "model": "gpt-5.6-sol",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 6},
            },
        }
        result = map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 2
        assert result.usage.cache_creation_tokens == 6

    def test_with_reasoning_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {
            "id": "r",
            "model": "o3",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 4}},
        }
        result = map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.reasoning_tokens == 4

    def test_none_usage(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"id": "r", "model": "gpt-4o", "output": []}
        result = map_responses_response(WireResponsesResponse.model_validate(response), "openai", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None
