"""Tests for Google Vertex AI type mappers."""

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest

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
from lmux_gcp_vertex._mappers import (
    map_embed_content_response,
    map_generate_content_chunk,
    map_generate_content_response,
    map_messages,
    map_response_format,
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


def _make_response(  # noqa: PLR0913
    *,
    text: str | None = None,
    function_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = "STOP",
    prompt_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int | None = None,
    thoughts: list[str] | None = None,
) -> MagicMock:
    """Create a mock GenerateContentResponse."""
    response = MagicMock()
    candidate = MagicMock()

    parts = []
    if thoughts:
        for thought_text in thoughts:
            thought_part = MagicMock()
            thought_part.thought = True
            thought_part.text = thought_text
            thought_part.function_call = None
            parts.append(thought_part)
    if text is not None:
        text_part = MagicMock()
        text_part.thought = False
        text_part.text = text
        text_part.function_call = None
        parts.append(text_part)
    if function_calls:
        for fc in function_calls:
            fc_part = MagicMock()
            fc_part.thought = False
            fc_part.text = None
            fc_part.function_call = MagicMock()
            fc_part.function_call.id = fc.get("id")
            fc_part.function_call.name = fc.get("name")
            fc_part.function_call.args = fc.get("args")
            parts.append(fc_part)

    content = MagicMock()
    content.parts = parts or None
    candidate.content = content if parts else None
    candidate.finish_reason = MagicMock(value=finish_reason) if finish_reason else None

    response.candidates = [candidate]

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = output_tokens
    usage.cached_content_token_count = cached_tokens
    usage.thoughts_token_count = None
    response.usage_metadata = usage

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
        system, contents = map_messages(
            [
                SystemMessage(content="Be helpful."),
                DeveloperMessage(content="Be concise."),
            ]
        )
        assert system == "Be helpful.\nBe concise."
        assert contents == []

    def test_user_message_text(self) -> None:
        system, contents = map_messages([UserMessage(content="Hello")])
        assert system is None
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_user_message_multimodal_base64(self) -> None:
        raw_bytes = b"\x89PNG\r\n"
        b64_data = base64.b64encode(raw_bytes).decode()
        data_uri = f"data:image/png;base64,{b64_data}"
        parts: list[ContentPart] = [TextContent(text="What is this?"), ImageContent(url=data_uri)]
        system, contents = map_messages([UserMessage(content=parts)])

        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "What is this?"},
                    {"inline_data": {"data": raw_bytes, "mime_type": "image/png"}},
                ],
            }
        ]

    def test_user_message_multimodal_url(self) -> None:
        parts: list[ContentPart] = [ImageContent(url="https://example.com/image.png")]
        system, contents = map_messages([UserMessage(content=parts)])

        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [{"file_data": {"file_uri": "https://example.com/image.png", "mime_type": "image/*"}}],
            }
        ]

    def test_assistant_message_text(self) -> None:
        system, contents = map_messages([AssistantMessage(content="Hi!")])
        assert system is None
        assert contents == [{"role": "model", "parts": [{"text": "Hi!"}]}]

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        system, contents = map_messages([AssistantMessage(content="Let me check.", tool_calls=[tc])])

        assert system is None
        assert contents == [
            {
                "role": "model",
                "parts": [
                    {"text": "Let me check."},
                    {"function_call": {"id": "tc1", "name": "get_weather", "args": {"city": "NYC"}}},
                ],
            }
        ]

    def test_assistant_message_tool_calls_no_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        system, contents = map_messages([AssistantMessage(content=None, tool_calls=[tc])])

        assert system is None
        assert contents == [
            {
                "role": "model",
                "parts": [
                    {"function_call": {"id": "tc1", "name": "f", "args": {}}},
                ],
            }
        ]

    def test_tool_message_json_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="get_weather", arguments="{}"))
        system, contents = map_messages(
            [
                AssistantMessage(content=None, tool_calls=[tc]),
                ToolMessage(content='{"temperature": "72F"}', tool_call_id="tc1"),
            ]
        )

        assert system is None
        assert len(contents) == 2
        assert contents[1] == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "tc1",
                        "name": "get_weather",
                        "response": {"temperature": "72F"},
                    }
                }
            ],
        }

    def test_tool_message_plain_text_content(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="search", arguments="{}"))
        _system, contents = map_messages(
            [
                AssistantMessage(content=None, tool_calls=[tc]),
                ToolMessage(content="not json", tool_call_id="tc1"),
            ]
        )

        assert contents[1] == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "tc1",
                        "name": "search",
                        "response": {"result": "not json"},
                    }
                }
            ],
        }

    def test_tool_message_unknown_id_uses_id_as_name(self) -> None:
        _system, contents = map_messages(
            [
                ToolMessage(content='{"result": "ok"}', tool_call_id="unknown_id"),
            ]
        )

        assert contents[0] == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "unknown_id",
                        "name": "unknown_id",
                        "response": {"result": "ok"},
                    }
                }
            ],
        }

    def test_no_system_returns_none(self) -> None:
        system, _contents = map_messages([UserMessage(content="hi")])
        assert system is None

    def test_mixed_messages(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        system, contents = map_messages(msgs)
        assert system == "sys"
        assert contents == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]


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
        assert result == [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters_json_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            }
        ]

    def test_minimal_tool(self) -> None:
        tools = [Tool(function=FunctionDefinition(name="noop"))]
        result = map_tools(tools)
        assert result == [{"function_declarations": [{"name": "noop"}]}]


# MARK: map_response_format


class TestMapResponseFormat:
    def test_text_format(self) -> None:
        mime_type, schema = map_response_format(TextResponseFormat())
        assert mime_type is None
        assert schema is None

    def test_json_object_format(self) -> None:
        mime_type, schema = map_response_format(JsonObjectResponseFormat())
        assert mime_type == "application/json"
        assert schema is None

    def test_json_schema_format(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        mime_type, schema = map_response_format(rf)
        assert mime_type == "application/json"
        assert schema == {"type": "object"}


# MARK: map_generate_content_response


class TestMapGenerateContentResponse:
    def test_text_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hello!")
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gemini-2.0-flash",
            provider="gcp-vertex",
            finish_reason="stop",
        )

    def test_tool_call_response(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(
            function_calls=[{"id": "call_0", "name": "get_weather", "args": {"city": "NYC"}}],
            finish_reason="STOP",
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result == ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_0",
                    function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'),
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gemini-2.0-flash",
            provider="gcp-vertex",
            finish_reason="tool_calls",
        )

    def test_function_call_without_id(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(
            function_calls=[{"id": None, "name": "search", "args": {}}],
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.tool_calls is not None
        # ID should be auto-generated as call_{index}
        assert result.tool_calls[0].id.startswith("call_")

    def test_function_call_without_args(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(
            function_calls=[{"id": "c1", "name": "ping", "args": None}],
        )
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.arguments == "{}"

    def test_no_candidates(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        response.candidates = None
        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 0
        usage.cached_content_token_count = None
        response.usage_metadata = usage

        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None

    def test_empty_candidates(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        response.candidates = []
        response.usage_metadata = None

        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.content is None
        assert result.usage is None
        assert result.cost is None

    def test_cache_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="cached", cached_tokens=50)
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_no_usage(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hi")
        response.usage_metadata = None
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None

    def test_cost_none_for_unknown_model(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hi")
        result = map_generate_content_response(response, "unknown-model", "gcp-vertex", none_cost_fn)
        assert result.cost is None

    def test_safety_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text=None, finish_reason="SAFETY")
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.finish_reason == "content_filter"

    def test_max_tokens_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="truncated", finish_reason="MAX_TOKENS")
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.finish_reason == "length"

    def test_none_finish_reason(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hi", finish_reason=None)
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.finish_reason is None

    def test_unknown_finish_reason_passthrough(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hi", finish_reason="SOME_NEW_REASON")
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.finish_reason == "SOME_NEW_REASON"

    def test_thought_parts_extracted(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Answer", thoughts=["Thinking..."])
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning == "Thinking..."

    def test_thought_part_with_none_text(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        """A thought part with thought=True but text=None should be skipped without error."""
        response = MagicMock()
        candidate = MagicMock()

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.text = None
        thought_part.function_call = None

        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "Answer"
        text_part.function_call = None

        content = MagicMock()
        content.parts = [thought_part, text_part]
        candidate.content = content
        candidate.finish_reason = MagicMock(value="STOP")

        response.candidates = [candidate]
        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 5
        usage.cached_content_token_count = None
        usage.thoughts_token_count = None
        response.usage_metadata = usage

        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.content == "Answer"
        assert result.reasoning is None

    def test_no_content_on_candidate(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = _make_response(text="Hi")
        response.candidates[0].content = None
        result = map_generate_content_response(response, "gemini-2.0-flash", "gcp-vertex", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls is None


# MARK: map_generate_content_chunk


class TestMapGenerateContentChunk:
    def test_text_chunk(self) -> None:
        chunk = _make_response(text="Hello", finish_reason=None)
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result == ChatChunk(delta="Hello", model="gemini-2.0-flash")

    def test_function_call_chunk(self) -> None:
        chunk = _make_response(
            function_calls=[{"id": "call_0", "name": "get_weather", "args": {"city": "NYC"}}],
            finish_reason=None,
        )
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.tool_call_deltas is not None
        assert result.tool_call_deltas == [
            ToolCallDelta(
                index=0,
                id="call_0",
                type="function",
                function=FunctionCallDelta(name="get_weather", arguments='{"city": "NYC"}'),
            )
        ]

    def test_finish_reason_chunk(self) -> None:
        chunk = _make_response(finish_reason="STOP")
        chunk.candidates[0].content = None
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.finish_reason == "stop"

    def test_usage_chunk(self) -> None:
        chunk = _make_response(text=None, finish_reason=None)
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.usage == Usage(input_tokens=10, output_tokens=5)

    def test_empty_chunk(self) -> None:
        chunk = MagicMock()
        chunk.candidates = None
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result == ChatChunk(model="gemini-2.0-flash")

    def test_thought_parts_extracted_in_chunk(self) -> None:
        chunk = _make_response(text=None, thoughts=["Thinking..."], finish_reason=None)
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.delta is None
        assert result.reasoning_delta == "Thinking..."

    def test_thought_part_with_none_text_in_chunk(self) -> None:
        """A thought part with thought=True but text=None should be skipped in streaming."""
        chunk = MagicMock()
        candidate = MagicMock()

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.text = None
        thought_part.function_call = None

        content = MagicMock()
        content.parts = [thought_part]
        candidate.content = content
        candidate.finish_reason = None

        chunk.candidates = [candidate]
        chunk.usage_metadata = None

        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.delta is None
        assert result.reasoning_delta is None

    def test_function_call_without_args_in_chunk(self) -> None:
        chunk = _make_response(
            function_calls=[{"id": "c1", "name": "ping", "args": None}],
            finish_reason=None,
        )
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.tool_call_deltas is not None
        assert result.tool_call_deltas[0].function is not None
        assert result.tool_call_deltas[0].function.arguments == "{}"

    def test_chunk_with_tool_calls_has_tool_calls_finish_reason(self) -> None:
        chunk = _make_response(
            function_calls=[{"id": "c1", "name": "f", "args": {}}],
            finish_reason="STOP",
        )
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.finish_reason == "tool_calls"

    def test_nonterminal_tool_call_chunk_preserves_null_finish_reason(self) -> None:
        chunk = _make_response(
            function_calls=[{"id": "c1", "name": "f", "args": {}}],
            finish_reason=None,
        )
        chunk.usage_metadata = None
        result = map_generate_content_chunk(chunk, "gemini-2.0-flash")
        assert result.finish_reason is None


# MARK: map_embed_content_response


class TestMapEmbedContentResponse:
    def test_single_embedding(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        emb = MagicMock()
        emb.values = [0.1, 0.2, 0.3]
        response.embeddings = [emb]
        response.metadata = None

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=0, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="text-embedding-005",
            provider="gcp-vertex",
        )

    def test_multiple_embeddings(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        emb1 = MagicMock()
        emb1.values = [0.1, 0.2]
        emb2 = MagicMock()
        emb2.values = [0.3, 0.4]
        response.embeddings = [emb1, emb2]
        response.metadata = None

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]

    def test_empty_embeddings(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        response.embeddings = None
        response.metadata = None

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result.embeddings == []

    def test_embedding_with_none_values(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        emb = MagicMock()
        emb.values = None
        response.embeddings = [emb]
        response.metadata = None

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result.embeddings == [[]]

    def test_cost_none_for_unknown(self, none_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        response.embeddings = [MagicMock(values=[0.1])]
        response.metadata = None

        result = map_embed_content_response(response, "unknown-model", "gcp-vertex", none_cost_fn)
        assert result.cost is None

    def test_approximates_tokens_from_billable_characters(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        emb = MagicMock()
        emb.values = [0.1]
        response.embeddings = [emb]
        metadata = MagicMock()
        metadata.billable_character_count = 400
        response.metadata = metadata

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result.usage == Usage(input_tokens=100, output_tokens=0)

    def test_billable_character_count_none_falls_back_to_zero(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = MagicMock()
        emb = MagicMock()
        emb.values = [0.1]
        response.embeddings = [emb]
        metadata = MagicMock()
        metadata.billable_character_count = None
        response.metadata = metadata

        result = map_embed_content_response(response, "text-embedding-005", "gcp-vertex", noop_cost_fn)
        assert result.usage == Usage(input_tokens=0, output_tokens=0)
