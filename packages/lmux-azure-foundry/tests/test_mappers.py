"""Tests for Azure AI Foundry JSON mappers."""

from typing import Any

import pytest

from lmux.exceptions import InvalidRequestError
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
    ProviderContinuation,
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
    ResponsesStreamMapper,
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_messages_to_response_input,
    map_response_format,
    map_response_input,
    map_response_text_format,
    map_response_tool_choice,
    map_response_tools,
    map_responses_chat,
    map_responses_response,
    map_tool_choice,
    map_tools,
)
from lmux_azure_foundry._wire import (
    WireChunk,
    WireCompletion,
    WireEmbeddingResponse,
    WireIncompleteDetails,
    WireOutputItem,
    WireResponsesResponse,
    WireResponsesStreamEvent,
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

    def test_structured_message_maps_and_drops_cache_points(self) -> None:
        items = [
            ResponseInputMessage(
                role="user",
                content=[
                    TextContent(text="Describe this image"),
                    CachePointContent(),
                    ImageContent(url="https://example.com/image.png", detail="high"),
                ],
            )
        ]
        assert map_response_input(items) == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "high"},
                ],
            }
        ]

    def test_cache_point_only_message_is_dropped(self) -> None:
        items = [
            ResponseInputMessage(role="developer", content="Instructions"),
            ResponseInputMessage(role="user", content=[CachePointContent()]),
        ]
        assert map_response_input(items) == [{"role": "developer", "content": "Instructions"}]


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


# MARK: Responses API output items


@pytest.fixture
def reasoning_item() -> dict[str, Any]:
    return {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []}


@pytest.fixture
def call_item() -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lab",
        "arguments": '{"analyte":"LDL"}',
        "status": "completed",
    }


# MARK: map_messages_to_response_input


class TestMapMessagesToResponseInput:
    def test_system_developer_user_and_tool(self) -> None:
        result = map_messages_to_response_input(
            [
                SystemMessage(content="Be terse."),
                DeveloperMessage(content="Dev."),
                UserMessage(content="Hi"),
                ToolMessage(content="162 mg/dL", tool_call_id="call_1"),
            ]
        )
        assert result == [
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Dev."},
            {"role": "user", "content": "Hi"},
            {"type": "function_call_output", "call_id": "call_1", "output": "162 mg/dL"},
        ]

    def test_user_parts_drop_cache_points(self) -> None:
        result = map_messages_to_response_input(
            [
                UserMessage(
                    content=[TextContent(text="Look"), CachePointContent(), ImageContent(url="https://x/i.png")]
                ),
                UserMessage(content=[CachePointContent()]),
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look"},
                    {"type": "input_image", "image_url": "https://x/i.png", "detail": "auto"},
                ],
            }
        ]

    def test_assistant_without_continuation_is_rebuilt(self) -> None:
        tc = ToolCall(id="call_1", function=FunctionCallResult(name="lab", arguments="{}"))
        result = map_messages_to_response_input(
            [
                AssistantMessage(content="Checking.", tool_calls=[tc]),
                AssistantMessage(content=None, tool_calls=None),
            ]
        )
        assert result == [
            {"role": "assistant", "content": "Checking."},
            {"type": "function_call", "call_id": "call_1", "name": "lab", "arguments": "{}"},
        ]

    def test_assistant_with_continuation_replays_output(
        self, reasoning_item: dict[str, Any], call_item: dict[str, Any]
    ) -> None:
        continuation = ProviderContinuation(
            namespace="lmux_azure_foundry.responses", data={"output": [reasoning_item, call_item]}
        )
        tc = ToolCall(id="call_1", function=FunctionCallResult(name="lab", arguments="{}"))
        result = map_messages_to_response_input([AssistantMessage(tool_calls=[tc], continuation=continuation)])
        assert result == [reasoning_item, call_item]

    def test_replay_does_not_alias_the_continuation(self, reasoning_item: dict[str, Any]) -> None:
        continuation = ProviderContinuation(namespace="lmux_azure_foundry.responses", data={"output": [reasoning_item]})
        result = map_messages_to_response_input([AssistantMessage(continuation=continuation)])
        result[0]["summary"].append("mutated")
        assert continuation.data == {"output": [reasoning_item]}

    def test_foreign_continuation_is_ignored(self) -> None:
        continuation = ProviderContinuation(namespace="lmux_anthropic.messages", data={"content": []})
        result = map_messages_to_response_input([AssistantMessage(content="Hi", continuation=continuation)])
        assert result == [{"role": "assistant", "content": "Hi"}]

    def test_malformed_continuation_raises(self) -> None:
        continuation = ProviderContinuation(namespace="lmux_azure_foundry.responses", data={"output": ["not an item"]})
        with pytest.raises(InvalidRequestError, match="list of output items"):
            map_messages_to_response_input([AssistantMessage(continuation=continuation)])


# MARK: map_response_tools


class TestMapResponseTools:
    def test_flat_function_tools_default_to_non_strict(self) -> None:
        tools = [
            Tool(function=FunctionDefinition(name="lab", description="Labs.", parameters={"type": "object"})),
            Tool(function=FunctionDefinition(name="ping", strict=True)),
        ]
        assert map_response_tools(tools) == [
            {
                "type": "function",
                "name": "lab",
                "strict": False,
                "description": "Labs.",
                "parameters": {"type": "object"},
            },
            {"type": "function", "name": "ping", "strict": True},
        ]


# MARK: map_response_tool_choice


class TestMapResponseToolChoice:
    def test_function(self) -> None:
        assert map_response_tool_choice(ToolChoiceFunction(name="lab")) == {"type": "function", "name": "lab"}

    def test_literal(self) -> None:
        assert map_response_tool_choice("required") == "required"


# MARK: map_response_text_format


class TestMapResponseTextFormat:
    def test_text_and_json_object(self) -> None:
        assert map_response_text_format(TextResponseFormat()) == {"type": "text"}
        assert map_response_text_format(JsonObjectResponseFormat()) == {"type": "json_object"}

    def test_json_schema(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="out", json_schema={"type": "object", "properties": {}}, description="Out.", strict=True
        )
        assert map_response_text_format(rf) == {
            "type": "json_schema",
            "name": "out",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "description": "Out.",
            "strict": True,
        }

    def test_json_schema_minimal(self) -> None:
        rf = JsonSchemaResponseFormat(name="out", json_schema={"type": "object"})
        assert map_response_text_format(rf) == {
            "type": "json_schema",
            "name": "out",
            "schema": {"type": "object", "additionalProperties": False},
        }


# MARK: map_responses_chat


class TestMapResponsesChat:
    def test_reasoning_and_tool_call_keep_continuation(
        self, reasoning_item: dict[str, Any], call_item: dict[str, Any]
    ) -> None:
        reasoning = {
            **reasoning_item,
            "summary": [
                {"type": "summary_text", "text": "Need LDL."},
                {"type": "summary_text"},
                {"type": "summary_text", "text": "Then HDL."},
            ],
        }
        wire = WireResponsesResponse.model_validate(
            {
                "id": "resp_1",
                "model": "gpt-6-luna",
                "status": "completed",
                "output": [reasoning, call_item, {"type": "function_call", "call_id": None, "name": "x"}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 3}},
            }
        )
        result = map_responses_chat(wire, "azure-foundry", _noop_cost)
        assert result == ChatResponse(
            content=None,
            reasoning="Need LDL.\n\nThen HDL.",
            tool_calls=[ToolCall(id="call_1", function=FunctionCallResult(name="lab", arguments='{"analyte":"LDL"}'))],
            usage=Usage(input_tokens=10, output_tokens=5, reasoning_tokens=3),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-6-luna",
            provider="azure-foundry",
            finish_reason="tool_calls",
            continuation=ProviderContinuation(
                namespace="lmux_azure_foundry.responses",
                data={"output": [reasoning, call_item, {"type": "function_call", "name": "x"}]},
            ),
        )

    def test_text_only_has_no_continuation(self) -> None:
        wire = WireResponsesResponse.model_validate(
            {
                "id": "resp_2",
                "model": "gpt-6-luna",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}],
            }
        )
        result = map_responses_chat(wire, "azure-foundry", _noop_cost)
        assert result == ChatResponse(
            content="Done.", usage=None, cost=None, model="gpt-6-luna", provider="azure-foundry", finish_reason="stop"
        )

    @pytest.mark.parametrize(
        ("details", "expected"),
        [
            (WireIncompleteDetails(reason="max_output_tokens"), "length"),
            (WireIncompleteDetails(reason="content_filter"), "content_filter"),
            (WireIncompleteDetails(reason="other"), "other"),
            (None, "incomplete"),
        ],
    )
    def test_incomplete_reasons(self, details: WireIncompleteDetails | None, expected: str) -> None:
        wire = WireResponsesResponse(id="r", model="m", status="incomplete", incomplete_details=details)
        result = map_responses_chat(wire, "azure-foundry", _none_cost)
        assert result == ChatResponse(
            content=None, usage=None, cost=None, model="m", provider="azure-foundry", finish_reason=expected
        )


# MARK: ResponsesStreamMapper.map_event


class TestResponsesStreamMapperMapEvent:
    def test_text_delta(self) -> None:
        mapper = ResponsesStreamMapper("azure-foundry", _noop_cost)
        event = WireResponsesStreamEvent(type="response.output_text.delta", delta="Hi")
        assert mapper.map_event(event) == ChatChunk(delta="Hi", provider="azure-foundry")

    def test_reasoning_summary_parts_are_separated(self) -> None:
        mapper = ResponsesStreamMapper("azure-foundry", _noop_cost)
        part_added = WireResponsesStreamEvent(type="response.reasoning_summary_part.added")
        chunks = [
            mapper.map_event(part_added),
            mapper.map_event(WireResponsesStreamEvent(type="response.reasoning_summary_text.delta", delta="One.")),
            mapper.map_event(part_added),
            mapper.map_event(WireResponsesStreamEvent(type="response.reasoning_summary_text.delta", delta="Two.")),
        ]
        assert chunks == [
            None,
            ChatChunk(reasoning_delta="One.", provider="azure-foundry"),
            ChatChunk(reasoning_delta="\n\n", provider="azure-foundry"),
            ChatChunk(reasoning_delta="Two.", provider="azure-foundry"),
        ]

    def test_function_calls_are_indexed_in_start_order(self) -> None:
        mapper = ResponsesStreamMapper("azure-foundry", _noop_cost)
        first = mapper.map_event(
            WireResponsesStreamEvent(
                type="response.output_item.added",
                output_index=1,
                item=WireOutputItem(type="function_call", call_id="call_a", name="lab", arguments=""),
            )
        )
        second = mapper.map_event(
            WireResponsesStreamEvent(
                type="response.output_item.added",
                output_index=2,
                item=WireOutputItem(type="function_call", call_id="call_b", name="lab", arguments="{}"),
            )
        )
        args = mapper.map_event(
            WireResponsesStreamEvent(type="response.function_call_arguments.delta", output_index=1, delta='{"a')
        )
        assert first == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, id="call_a", type="function", function=FunctionCallDelta(name="lab"))
            ],
            provider="azure-foundry",
        )
        assert second == ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=1, id="call_b", type="function", function=FunctionCallDelta(name="lab", arguments="{}")
                )
            ],
            provider="azure-foundry",
        )
        assert args == ChatChunk(
            tool_call_deltas=[ToolCallDelta(index=0, function=FunctionCallDelta(arguments='{"a'))],
            provider="azure-foundry",
        )

    @pytest.mark.parametrize(
        "event",
        [
            WireResponsesStreamEvent(type="response.created"),
            WireResponsesStreamEvent(type="response.completed"),
            WireResponsesStreamEvent(
                type="response.output_item.added", output_index=0, item=WireOutputItem(type="reasoning")
            ),
            WireResponsesStreamEvent(type="response.output_item.added", output_index=0),
            WireResponsesStreamEvent(type="response.function_call_arguments.delta", output_index=7, delta="x"),
        ],
    )
    def test_ignored_events(self, event: WireResponsesStreamEvent) -> None:
        mapper = ResponsesStreamMapper("azure-foundry", _noop_cost)
        assert mapper.map_event(event) is None

    def test_terminal_event_carries_usage_cost_and_continuation(self, reasoning_item: dict[str, Any]) -> None:
        mapper = ResponsesStreamMapper("azure-foundry", _noop_cost)
        event = WireResponsesStreamEvent.model_validate(
            {
                "type": "response.incomplete",
                "response": {
                    "id": "r",
                    "model": "gpt-6-luna",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [reasoning_item],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }
        )
        assert mapper.map_event(event) == ChatChunk(
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            finish_reason="length",
            model="gpt-6-luna",
            provider="azure-foundry",
            continuation=ProviderContinuation(
                namespace="lmux_azure_foundry.responses", data={"output": [reasoning_item]}
            ),
        )
