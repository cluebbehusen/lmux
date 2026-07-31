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
    has_cache_breakpoint,
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_response_input,
    map_responses_response,
    map_tool_choice,
    map_tools,
    supports_explicit_prompt_cache,
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

    # --- explicit prompt caching (gpt-5.6+): map cache points to prompt_cache_breakpoint ---

    def test_explicit_cache_attaches_to_preceding_block(self) -> None:
        result = map_messages(
            [UserMessage(content=[TextContent(text="Big"), CachePointContent()])], explicit_cache=True
        )
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Big", "prompt_cache_breakpoint": {"mode": "explicit"}}],
            }
        ]

    def test_explicit_cache_mid_message_marks_only_preceding_block(self) -> None:
        result = map_messages(
            [UserMessage(content=[TextContent(text="Stable"), CachePointContent(), TextContent(text="Varying")])],
            explicit_cache=True,
        )
        assert result[0]["content"] == [
            {"type": "text", "text": "Stable", "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "text", "text": "Varying"},
        ]

    def test_explicit_cache_first_breakpoint_wins_within_message(self) -> None:
        result = map_messages(
            [UserMessage(content=[TextContent(text="ctx"), CachePointContent(), CachePointContent()])],
            explicit_cache=True,
        )
        assert result[0]["content"] == [
            {"type": "text", "text": "ctx", "prompt_cache_breakpoint": {"mode": "explicit"}}
        ]

    def test_explicit_cache_ttl_is_ignored(self) -> None:
        # OpenAI's ttl is request-wide (prompt_cache_options.ttl, only "30m"); per-breakpoint ttl doesn't map.
        result = map_messages(
            [UserMessage(content=[TextContent(text="ctx"), CachePointContent(ttl="1h")])], explicit_cache=True
        )
        assert result[0]["content"] == [
            {"type": "text", "text": "ctx", "prompt_cache_breakpoint": {"mode": "explicit"}}
        ]

    def test_explicit_leading_cache_attaches_to_previous_string_message(self) -> None:
        result = map_messages(
            [UserMessage(content="Hello"), UserMessage(content=[CachePointContent()])], explicit_cache=True
        )
        # Prior string content is promoted to a text block carrying the breakpoint; the marker-only message is dropped.
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "prompt_cache_breakpoint": {"mode": "explicit"}}],
            }
        ]

    def test_explicit_leading_cache_attaches_to_previous_blocks(self) -> None:
        result = map_messages(
            [
                UserMessage(content=[TextContent(text="A")]),
                UserMessage(content=[CachePointContent(), TextContent(text="B")]),
            ],
            explicit_cache=True,
        )
        assert result[0]["content"] == [{"type": "text", "text": "A", "prompt_cache_breakpoint": {"mode": "explicit"}}]
        assert result[1]["content"] == [{"type": "text", "text": "B"}]

    def test_explicit_leading_cache_with_no_prefix_is_dropped(self) -> None:
        result = map_messages([UserMessage(content=[CachePointContent(), TextContent(text="Hi")])], explicit_cache=True)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_explicit_first_leading_cache_wins(self) -> None:
        result = map_messages(
            [
                UserMessage(content="prev"),
                UserMessage(content=[CachePointContent(), CachePointContent(), TextContent(text="Hi")]),
            ],
            explicit_cache=True,
        )
        assert result[0]["content"] == [
            {"type": "text", "text": "prev", "prompt_cache_breakpoint": {"mode": "explicit"}}
        ]

    def test_explicit_leading_cache_prior_already_marked_is_noop(self) -> None:
        result = map_messages(
            [
                UserMessage(content=[TextContent(text="A"), CachePointContent()]),
                UserMessage(content=[CachePointContent()]),
            ],
            explicit_cache=True,
        )
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "A", "prompt_cache_breakpoint": {"mode": "explicit"}}],
            }
        ]

    def test_explicit_leading_cache_prior_no_content_is_noop(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        result = map_messages(
            [AssistantMessage(tool_calls=[tc]), UserMessage(content=[CachePointContent()])], explicit_cache=True
        )
        assert "content" not in result[0]

    def test_explicit_leading_cache_prior_empty_string_is_noop(self) -> None:
        result = map_messages(
            [UserMessage(content=""), UserMessage(content=[CachePointContent()])], explicit_cache=True
        )
        assert result == [{"role": "user", "content": ""}]

    def test_explicit_leading_cache_prior_empty_blocks_is_noop(self) -> None:
        result = map_messages(
            [UserMessage(content=[]), UserMessage(content=[CachePointContent()])], explicit_cache=True
        )
        assert result == [{"role": "user", "content": []}]


class TestSupportsExplicitPromptCache:
    def test_gpt_5_6_supported(self) -> None:
        assert supports_explicit_prompt_cache("gpt-5.6-sol") is True

    def test_older_model_not_supported(self) -> None:
        assert supports_explicit_prompt_cache("gpt-5.5") is False

    def test_later_generation_supported(self) -> None:
        assert supports_explicit_prompt_cache("gpt-6") is True

    def test_non_gpt_model_not_supported(self) -> None:
        assert supports_explicit_prompt_cache("o4-mini") is False


class TestHasCacheBreakpoint:
    def test_true_when_a_block_carries_a_breakpoint(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "x", "prompt_cache_breakpoint": {}}]}]
        assert has_cache_breakpoint(messages) is True

    def test_false_when_no_breakpoint(self) -> None:
        assert has_cache_breakpoint([{"role": "user", "content": [{"type": "text", "text": "x"}]}]) is False

    def test_false_for_string_content(self) -> None:
        assert has_cache_breakpoint([{"role": "user", "content": "hi"}]) is False


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

    def test_explicit_cache_maps_text_and_image_parts(self) -> None:
        items = [
            ResponseInputMessage(
                role="user",
                content=[
                    TextContent(text="Stable"),
                    ImageContent(url="https://example.com/image.png", detail="high"),
                    CachePointContent(),
                    TextContent(text="Variable"),
                ],
            )
        ]
        assert map_response_input(items, explicit_cache=True) == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Stable"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.png",
                        "detail": "high",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    },
                    {"type": "input_text", "text": "Variable"},
                ],
            }
        ]

    def test_explicit_leading_cache_marks_previous_message(self) -> None:
        items = [
            ResponseInputMessage(role="developer", content="Stable instructions"),
            ResponseInputMessage(role="user", content=[CachePointContent(), TextContent(text="Question")]),
        ]
        assert map_response_input(items, explicit_cache=True) == [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stable instructions",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "Question"}]},
        ]

    def test_cache_points_dropped_without_explicit_support(self) -> None:
        items = [ResponseInputMessage(role="user", content=[TextContent(text="Stable"), CachePointContent()])]
        assert map_response_input(items) == [{"role": "user", "content": [{"type": "input_text", "text": "Stable"}]}]

    def test_marker_only_message_after_function_item_is_dropped(self) -> None:
        items = [
            ResponseInputFunctionCallOutput(call_id="call_1", output="result"),
            ResponseInputMessage(role="user", content=[CachePointContent()]),
        ]
        assert map_response_input(items, explicit_cache=True) == [
            {"type": "function_call_output", "call_id": "call_1", "output": "result"}
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
