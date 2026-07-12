"""Tests for Azure AI Foundry JSON mappers."""

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
    ResponseInputFunctionCall,
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
from lmux_azure_foundry._mappers import (
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
from lmux_azure_foundry._wire import (
    WireChunk,
    WireCompletion,
    WireEmbeddingResponse,
    WireResponsesResponse,
)


def _noop_cost(_model: str, _usage: Usage) -> Cost | None:
    return Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0)


def _none_cost(_model: str, _usage: Usage) -> Cost | None:
    return None


# MARK: map_messages


class TestMapMessages:
    def test_system(self) -> None:
        assert map_messages([SystemMessage(content="s")]) == [{"role": "system", "content": "s"}]

    def test_developer(self) -> None:
        assert map_messages([DeveloperMessage(content="d")]) == [{"role": "developer", "content": "d"}]

    def test_user_text(self) -> None:
        assert map_messages([UserMessage(content="hi")]) == [{"role": "user", "content": "hi"}]

    def test_user_content_parts(self) -> None:
        parts: list[ContentPart] = [TextContent(text="t"), ImageContent(url="http://x", detail="high")]
        assert map_messages([UserMessage(content=parts)]) == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "t"},
                    {"type": "image_url", "image_url": {"url": "http://x", "detail": "high"}},
                ],
            }
        ]

    def test_originally_empty_content_list_forwarded(self) -> None:
        assert map_messages([UserMessage(content=[])]) == [{"role": "user", "content": []}]

    def test_cache_points_dropped(self) -> None:
        assert map_messages([UserMessage(content=[TextContent(text="Hi"), CachePointContent()])]) == [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        ]

    def test_marker_only_message_skipped(self) -> None:
        assert map_messages([UserMessage(content="Hello"), UserMessage(content=[CachePointContent()])]) == [
            {"role": "user", "content": "Hello"}
        ]

    def test_assistant_content(self) -> None:
        assert map_messages([AssistantMessage(content="a")]) == [{"role": "assistant", "content": "a"}]

    def test_assistant_tool_calls(self) -> None:
        msg = AssistantMessage(tool_calls=[ToolCall(id="c1", function=FunctionCallResult(name="f", arguments="{}"))])
        assert map_messages([msg]) == [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            }
        ]

    def test_tool_message(self) -> None:
        assert map_messages([ToolMessage(content="r", tool_call_id="c1")]) == [
            {"role": "tool", "content": "r", "tool_call_id": "c1"}
        ]

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
    def test_minimal(self) -> None:
        assert map_tools([Tool(function=FunctionDefinition(name="f"))]) == [
            {"type": "function", "function": {"name": "f"}}
        ]

    def test_full(self) -> None:
        tool = Tool(function=FunctionDefinition(name="f", description="d", parameters={"type": "object"}, strict=True))
        assert map_tools([tool]) == [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {"type": "object"}, "strict": True},
            }
        ]


# MARK: map_tool_choice


class TestMapToolChoice:
    def test_auto(self) -> None:
        assert map_tool_choice("auto") == "auto"

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
            "json_schema": {"name": "test", "schema": {"type": "object", "additionalProperties": False}},
        }

    def test_json_schema_full(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"}, description="A test", strict=True)
        assert map_response_format(rf) == {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "additionalProperties": False},
                "description": "A test",
                "strict": True,
            },
        }


# MARK: map_response_input


class TestMapResponseInput:
    def test_string(self) -> None:
        assert map_response_input("hello") == "hello"

    def test_items(self) -> None:
        items = [
            ResponseInputMessage(role="user", content="hi"),
            ResponseInputFunctionCall(call_id="c1", name="f", arguments="{}"),
        ]
        assert map_response_input(items) == [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        ]


# MARK: map_chat_completion


class TestMapChatCompletion:
    def test_basic(self) -> None:
        completion = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        assert map_chat_completion(
            WireCompletion.model_validate(completion), "azure-foundry", _noop_cost
        ) == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-4o",
            provider="azure-foundry",
            finish_reason="stop",
        )

    def test_with_tool_calls(self) -> None:
        completion = {
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
                            {"id": "tc2", "type": "other", "function": {"name": "g", "arguments": "{}"}},
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "azure-foundry", _noop_cost)
        assert result.content is None
        assert result.tool_calls == [ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments='{"x": 1}'))]

    def test_reasoning_content(self) -> None:
        completion = {
            "model": "o3",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "A", "reasoning_content": "because"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "azure-foundry", _noop_cost)
        assert result.reasoning == "because"

    def test_cache_and_reasoning_tokens(self) -> None:
        completion = {
            "model": "o3",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 50},
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "azure-foundry", _noop_cost)
        assert result.usage == Usage(input_tokens=10, output_tokens=25, cache_read_tokens=50, reasoning_tokens=20)

    def test_none_usage(self) -> None:
        completion = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}],
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "azure-foundry", _noop_cost)
        assert result.usage is None
        assert result.cost is None

    def test_cost_none_when_unknown(self) -> None:
        completion = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = map_chat_completion(WireCompletion.model_validate(completion), "azure-foundry", _none_cost)
        assert result.cost is None


# MARK: map_chat_chunk


class TestMapChatChunk:
    def test_content_delta(self) -> None:
        chunk = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hello"}}]}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry") == ChatChunk(
            delta="Hello", model="gpt-4o", provider="azure-foundry"
        )

    def test_reasoning_delta(self) -> None:
        chunk = {"model": "o3", "choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry").reasoning_delta == "hmm"

    def test_tool_call_delta(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "tc1", "type": "function", "function": {"name": "f", "arguments": '{"x'}}
                        ]
                    },
                }
            ],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry").tool_call_deltas == [
            ToolCallDelta(index=0, id="tc1", type="function", function=FunctionCallDelta(name="f", arguments='{"x'))
        ]

    def test_tool_call_delta_without_function(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "tc1", "type": "other"}]}}],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry").tool_call_deltas == [
            ToolCallDelta(index=0, id="tc1", type=None, function=None)
        ]

    def test_final_chunk_with_usage(self) -> None:
        chunk = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry") == ChatChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="gpt-4o",
            provider="azure-foundry",
        )

    def test_empty_choices(self) -> None:
        chunk = {"model": "gpt-4o", "choices": []}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "azure-foundry") == ChatChunk(
            model="gpt-4o", provider="azure-foundry"
        )


# MARK: map_embedding_response


class TestMapEmbeddingResponse:
    def test_single(self) -> None:
        response = {
            "model": "text-embedding-3-small",
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 5},
        }
        assert map_embedding_response(
            WireEmbeddingResponse.model_validate(response), "azure-foundry", _noop_cost
        ) == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="text-embedding-3-small",
            provider="azure-foundry",
        )

    def test_multiple_sorted_by_index(self) -> None:
        response = {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {"prompt_tokens": 10},
        }
        assert map_embedding_response(
            WireEmbeddingResponse.model_validate(response), "azure-foundry", _noop_cost
        ).embeddings == [[0.1, 0.2], [0.3, 0.4]]


# MARK: map_responses_response


class TestMapResponsesResponse:
    def test_basic(self) -> None:
        response = {
            "id": "resp_1",
            "model": "gpt-5-pro",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        assert map_responses_response(
            WireResponsesResponse.model_validate(response), "azure-foundry", _noop_cost
        ) == ResponseResponse(
            id="resp_1",
            output_text="Hi!",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-5-pro",
            provider="azure-foundry",
        )

    def test_output_text_aggregates_and_skips_non_text(self) -> None:
        response = {
            "id": "resp_1",
            "model": "gpt-5-pro",
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "skip"}]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hel"},
                        {"type": "refusal", "refusal": "no"},
                        {"type": "output_text", "text": "lo"},
                    ],
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        assert (
            map_responses_response(
                WireResponsesResponse.model_validate(response), "azure-foundry", _noop_cost
            ).output_text
            == "Hello"
        )

    def test_cache_and_reasoning_tokens(self) -> None:
        response = {
            "id": "resp_1",
            "model": "gpt-5-pro",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 4},
            },
        }
        result = map_responses_response(WireResponsesResponse.model_validate(response), "azure-foundry", _noop_cost)
        assert result.output_text == ""
        assert result.usage == Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3, reasoning_tokens=4)

    def test_no_usage(self) -> None:
        response = {"id": "resp_1", "model": "gpt-5-pro", "output": []}
        result = map_responses_response(WireResponsesResponse.model_validate(response), "azure-foundry", _noop_cost)
        assert result.usage is None
        assert result.cost is None
