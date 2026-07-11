"""Tests for Google (Gemini REST) type mappers."""

import base64
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
    ServerToolDelta,
    ServerToolResult,
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
from lmux_google._mappers import (
    Json,
    map_batch_embeddings_response,
    map_generate_content_chunk,
    map_generate_content_response,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
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


def _response(  # noqa: PLR0913
    *,
    parts: list[Json] | None = None,
    has_content: bool = True,
    finish_reason: str | None = "STOP",
    prompt_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int | None = None,
    usage: bool = True,
) -> Json:
    """Build a raw generateContent JSON body."""
    candidate: Json = {}
    if has_content:
        candidate["content"] = {"role": "model", "parts": parts or []}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    response: Json = {"candidates": [candidate]}
    if usage:
        meta: Json = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": output_tokens}
        if cached_tokens is not None:
            meta["cachedContentTokenCount"] = cached_tokens
        response["usageMetadata"] = meta
    return response


# MARK: map_messages


class TestMapMessages:
    def test_system_message(self) -> None:
        system, contents = map_messages([SystemMessage(content="Be helpful.")])
        assert system == "Be helpful."
        assert contents == []

    def test_developer_message(self) -> None:
        system, contents = map_messages([DeveloperMessage(content="Be concise.")])
        assert system == "Be concise."
        assert contents == []

    def test_multiple_system_messages_concatenated(self) -> None:
        system, _contents = map_messages(
            [SystemMessage(content="Be helpful."), DeveloperMessage(content="Be concise.")]
        )
        assert system == "Be helpful.\nBe concise."

    def test_user_message_text(self) -> None:
        system, contents = map_messages([UserMessage(content="Hello")])
        assert system is None
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_originally_empty_content_list_is_forwarded(self) -> None:
        _, contents = map_messages([UserMessage(content=[])])
        assert contents == [{"role": "user", "parts": []}]

    def test_user_message_multimodal_base64(self) -> None:
        b64_data = base64.b64encode(b"\x89PNG\r\n").decode()
        parts: list[ContentPart] = [
            TextContent(text="What is this?"),
            ImageContent(url=f"data:image/png;base64,{b64_data}"),
        ]
        _system, contents = map_messages([UserMessage(content=parts)])
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "What is this?"},
                    {"inlineData": {"data": b64_data, "mimeType": "image/png"}},
                ],
            }
        ]

    def test_user_message_multimodal_url(self) -> None:
        parts: list[ContentPart] = [ImageContent(url="https://example.com/image.png")]
        _system, contents = map_messages([UserMessage(content=parts)])
        assert contents == [
            {
                "role": "user",
                "parts": [{"fileData": {"fileUri": "https://example.com/image.png", "mimeType": "image/*"}}],
            }
        ]

    def test_assistant_message_text(self) -> None:
        _system, contents = map_messages([AssistantMessage(content="Hi!")])
        assert contents == [{"role": "model", "parts": [{"text": "Hi!"}]}]

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        _system, contents = map_messages([AssistantMessage(content="Let me check.", tool_calls=[tc])])
        assert contents == [
            {
                "role": "model",
                "parts": [
                    {"text": "Let me check."},
                    {"functionCall": {"id": "tc1", "name": "get_weather", "args": {"city": "NYC"}}},
                ],
            }
        ]

    def test_assistant_message_tool_calls_no_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        _system, contents = map_messages([AssistantMessage(content=None, tool_calls=[tc])])
        assert contents == [{"role": "model", "parts": [{"functionCall": {"id": "tc1", "name": "f", "args": {}}}]}]

    def test_tool_message_json_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="get_weather", arguments="{}"))
        _system, contents = map_messages(
            [
                AssistantMessage(content=None, tool_calls=[tc]),
                ToolMessage(content='{"temperature": "72F"}', tool_call_id="tc1"),
            ]
        )
        assert contents[1] == {
            "role": "user",
            "parts": [{"functionResponse": {"id": "tc1", "name": "get_weather", "response": {"temperature": "72F"}}}],
        }

    def test_tool_message_plain_text_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="search", arguments="{}"))
        _system, contents = map_messages(
            [
                AssistantMessage(content=None, tool_calls=[tc]),
                ToolMessage(content="not json", tool_call_id="tc1"),
            ]
        )
        assert contents[1]["parts"][0]["functionResponse"]["response"] == {"result": "not json"}

    def test_tool_message_unknown_id_uses_id_as_name(self) -> None:
        _system, contents = map_messages([ToolMessage(content='{"result": "ok"}', tool_call_id="unknown_id")])
        assert contents[0]["parts"][0]["functionResponse"]["name"] == "unknown_id"

    def test_no_system_returns_none(self) -> None:
        system, _contents = map_messages([UserMessage(content="hi")])
        assert system is None

    def test_mixed_messages(self) -> None:
        system, contents = map_messages(
            [SystemMessage(content="sys"), UserMessage(content="hi"), AssistantMessage(content="hello")]
        )
        assert system == "sys"
        assert contents == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]

    def test_cache_points_dropped(self) -> None:
        _, contents = map_messages([UserMessage(content=[TextContent(text="Hi"), CachePointContent()])])
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_marker_only_message_skipped(self) -> None:
        _, contents = map_messages([UserMessage(content="Hello"), UserMessage(content=[CachePointContent()])])
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]


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
        assert map_tools(tools) == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parametersJsonSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ]
            }
        ]

    def test_minimal_tool(self) -> None:
        assert map_tools([Tool(function=FunctionDefinition(name="noop"))]) == [
            {"functionDeclarations": [{"name": "noop"}]}
        ]


# MARK: map_response_format


class TestMapResponseFormat:
    def test_text_format(self) -> None:
        assert map_response_format(TextResponseFormat()) == (None, None)

    def test_json_object_format(self) -> None:
        assert map_response_format(JsonObjectResponseFormat()) == ("application/json", None)

    def test_json_schema_format(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        assert map_response_format(rf) == ("application/json", {"type": "object"})


# MARK: map_tool_choice


class TestMapToolChoice:
    def test_auto(self) -> None:
        assert map_tool_choice("auto") == {"functionCallingConfig": {"mode": "AUTO"}}

    def test_required(self) -> None:
        assert map_tool_choice("required") == {"functionCallingConfig": {"mode": "ANY"}}

    def test_none(self) -> None:
        assert map_tool_choice("none") == {"functionCallingConfig": {"mode": "NONE"}}

    def test_specific_function(self) -> None:
        assert map_tool_choice(ToolChoiceFunction(name="get_weather")) == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
        }


# MARK: map_generate_content_response


class TestMapGenerateContentResponse:
    def test_text_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hello!"}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gemini-2.0-flash",
            provider="google",
            finish_reason="stop",
        )

    def test_tool_call_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"functionCall": {"id": "call_0", "name": "get_weather", "args": {"city": "NYC"}}}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.tool_calls == [
            ToolCall(id="call_0", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        ]
        assert result.finish_reason == "tool_calls"

    def test_function_call_without_id(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"functionCall": {"name": "search", "args": {}}}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.tool_calls is not None
        assert result.tool_calls[0].id.startswith("call_")

    def test_function_call_without_args(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"functionCall": {"id": "c1", "name": "ping"}}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.arguments == "{}"

    def test_function_call_without_name(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"functionCall": {"id": "c1", "args": {}}}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == ""

    def test_no_candidates(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"candidates": None, "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0}}
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None
        assert result.usage is not None

    def test_empty_candidates(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_generate_content_response({"candidates": []}, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content is None
        assert result.usage is None
        assert result.cost is None

    def test_cache_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "cached"}], cached_tokens=50)
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_no_usage(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hi"}], usage=False)
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None

    def test_cost_none_for_unknown_model(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hi"}])
        result = map_generate_content_response(response, "unknown-model", "google", none_cost_fn)
        assert result.cost is None

    def test_safety_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[], finish_reason="SAFETY")
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.finish_reason == "content_filter"

    def test_max_tokens_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "truncated"}], finish_reason="MAX_TOKENS")
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.finish_reason == "length"

    def test_none_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hi"}], finish_reason=None)
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.finish_reason is None

    def test_unknown_finish_reason_passthrough(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hi"}], finish_reason="SOME_NEW_REASON")
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.finish_reason == "SOME_NEW_REASON"

    def test_thought_parts_extracted(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"thought": True, "text": "Thinking..."}, {"text": "Answer"}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning == "Thinking..."

    def test_thought_part_with_none_text(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"thought": True}, {"text": "Answer"}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning is None

    def test_no_content_on_candidate(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(has_content=False)
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None

    def test_code_execution_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(
            parts=[
                {"text": "The answer is 42."},
                {"executableCode": {"code": "print(42)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "42\n"}},
            ]
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content == "The answer is 42."
        assert result.server_tool_results == [
            ServerToolResult(
                name="code_execution",
                input={"code": "print(42)", "language": "PYTHON"},
                output="42\n",
                provider_specific_fields={"outcome": "OUTCOME_OK"},
            )
        ]

    def test_code_execution_without_text(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(
            parts=[
                {"executableCode": {"code": "print(42)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "42\n"}},
            ]
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.content is None
        assert result.server_tool_results is not None

    def test_code_execution_no_outcome(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(
            parts=[
                {"executableCode": {"code": "x = 1"}},
                {"codeExecutionResult": {}},
            ]
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.server_tool_results is not None
        assert result.server_tool_results[0].input == {"code": "x = 1", "language": None}
        assert result.server_tool_results[0].output is None
        assert result.server_tool_results[0].provider_specific_fields is None

    def test_multiple_code_executions(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(
            parts=[
                {"executableCode": {"code": "print(1)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}},
                {"executableCode": {"code": "print(2)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "2\n"}},
            ]
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.server_tool_results is not None
        assert [r.output for r in result.server_tool_results] == ["1\n", "2\n"]

    def test_no_code_execution_returns_none(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _response(parts=[{"text": "Hello!"}])
        result = map_generate_content_response(response, "gemini-2.0-flash", "google", noop_cost_fn)
        assert result.server_tool_results is None


# MARK: map_generate_content_chunk


class TestMapGenerateContentChunk:
    def test_text_chunk(self) -> None:
        chunk = _response(parts=[{"text": "Hello"}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result == ChatChunk(delta="Hello", model="gemini-2.0-flash", provider="google")

    def test_function_call_chunk(self) -> None:
        chunk = _response(
            parts=[{"functionCall": {"id": "call_0", "name": "get_weather", "args": {"city": "NYC"}}}],
            finish_reason=None,
            usage=False,
        )
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.tool_call_deltas == [
            ToolCallDelta(
                index=0,
                id="call_0",
                type="function",
                function=FunctionCallDelta(name="get_weather", arguments='{"city": "NYC"}'),
            )
        ]

    def test_function_call_chunk_without_id(self) -> None:
        chunk = _response(parts=[{"functionCall": {"name": "f", "args": {}}}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.tool_call_deltas is not None
        assert result.tool_call_deltas[0].id == "call_0"

    def test_finish_reason_chunk(self) -> None:
        chunk = _response(has_content=False, finish_reason="STOP", usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.finish_reason == "stop"

    def test_usage_chunk(self) -> None:
        chunk = _response(parts=[], finish_reason=None)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.usage == Usage(input_tokens=10, output_tokens=5)

    def test_empty_chunk(self) -> None:
        result = map_generate_content_chunk({"candidates": None}, "gemini-2.0-flash", "google")
        assert result == ChatChunk(model="gemini-2.0-flash", provider="google")

    def test_thought_parts_extracted_in_chunk(self) -> None:
        chunk = _response(parts=[{"thought": True, "text": "Thinking..."}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.delta is None
        assert result.reasoning_delta == "Thinking..."

    def test_thought_part_with_none_text_in_chunk(self) -> None:
        chunk = _response(parts=[{"thought": True}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.delta is None
        assert result.reasoning_delta is None

    def test_function_call_without_args_in_chunk(self) -> None:
        chunk = _response(parts=[{"functionCall": {"id": "c1", "name": "ping"}}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.tool_call_deltas is not None
        assert result.tool_call_deltas[0].function is not None
        assert result.tool_call_deltas[0].function.arguments == "{}"

    def test_chunk_with_tool_calls_has_tool_calls_finish_reason(self) -> None:
        chunk = _response(
            parts=[{"functionCall": {"id": "c1", "name": "f", "args": {}}}], finish_reason="STOP", usage=False
        )
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.finish_reason == "tool_calls"

    def test_nonterminal_tool_call_chunk_preserves_null_finish_reason(self) -> None:
        chunk = _response(
            parts=[{"functionCall": {"id": "c1", "name": "f", "args": {}}}], finish_reason=None, usage=False
        )
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.finish_reason is None

    def test_code_execution_chunk(self) -> None:
        chunk = _response(
            parts=[
                {"executableCode": {"code": "print(42)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "42\n"}},
            ],
            finish_reason=None,
            usage=False,
        )
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.server_tool_deltas == [
            ServerToolDelta(index=0, name="code_execution", input_delta='{"code": "print(42)", "language": "PYTHON"}'),
            ServerToolDelta(index=0, output_delta="42\n"),
        ]

    def test_code_execution_chunk_no_server_tool_deltas_when_absent(self) -> None:
        chunk = _response(parts=[{"text": "Hello"}], finish_reason=None, usage=False)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.server_tool_deltas is None

    def test_multiple_code_executions_in_chunk(self) -> None:
        chunk = _response(
            parts=[
                {"executableCode": {"code": "print(1)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}},
                {"executableCode": {"code": "print(2)", "language": "PYTHON"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "2\n"}},
            ],
            finish_reason=None,
            usage=False,
        )
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash", "google")
        assert result.server_tool_deltas is not None
        assert len(result.server_tool_deltas) == 4
        assert result.server_tool_deltas[1].index == 0
        assert result.server_tool_deltas[3].index == 1
        assert result.server_tool_deltas[3].output_delta == "2\n"


# MARK: map_batch_embeddings_response


class TestMapBatchEmbeddingsResponse:
    def test_single_embedding(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}
        result = map_batch_embeddings_response(response, "text-embedding-005", "google", noop_cost_fn)
        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=0, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="text-embedding-005",
            provider="google",
        )

    def test_multiple_embeddings(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]}
        result = map_batch_embeddings_response(response, "text-embedding-005", "google", noop_cost_fn)
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]

    def test_empty_embeddings(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_batch_embeddings_response({"embeddings": None}, "text-embedding-005", "google", noop_cost_fn)
        assert result.embeddings == []

    def test_embedding_with_none_values(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": None}]}
        result = map_batch_embeddings_response(response, "text-embedding-005", "google", noop_cost_fn)
        assert result.embeddings == [[]]

    def test_cost_none_for_unknown(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": [0.1]}]}
        result = map_batch_embeddings_response(response, "unknown-model", "google", none_cost_fn)
        assert result.cost is None

    def test_approximates_tokens_from_billable_characters(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": [0.1]}], "metadata": {"billableCharacterCount": 400}}
        result = map_batch_embeddings_response(response, "text-embedding-005", "google", noop_cost_fn)
        assert result.usage == Usage(input_tokens=100, output_tokens=0)

    def test_billable_character_count_none_falls_back_to_zero(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = {"embeddings": [{"values": [0.1]}], "metadata": {"billableCharacterCount": None}}
        result = map_batch_embeddings_response(response, "text-embedding-005", "google", noop_cost_fn)
        assert result.usage == Usage(input_tokens=0, output_tokens=0)
