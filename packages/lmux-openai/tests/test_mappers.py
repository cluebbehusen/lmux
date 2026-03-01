"""Tests for OpenAI type mappers."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_message_function_tool_call import Function as ToolCallFunction
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage as EmbUsage
from openai.types.embedding import Embedding

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCallUnion

from lmux.types import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
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


@pytest.fixture
def chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="Hello!", role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def responses_mock() -> MagicMock:
    mock = MagicMock()
    mock.id = "resp_123"
    mock.output_text = "Hello!"
    mock.model = "gpt-4o"
    mock.usage.input_tokens = 10
    mock.usage.output_tokens = 5
    mock.usage.input_tokens_details = None
    return mock


# MARK: map_messages


class TestMapMessages:
    def test_system_message(self) -> None:
        result = map_messages([SystemMessage(content="Be helpful.")])
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_developer_message(self) -> None:
        result = map_messages([DeveloperMessage(content="Be concise.")])
        assert result == [{"role": "developer", "content": "Be concise."}]

    def test_user_message_text(self) -> None:
        result = map_messages([UserMessage(content="Hello")])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_user_message_multimodal(self) -> None:
        parts = [TextContent(text="What?"), ImageContent(url="https://img.png", detail="high")]
        result = map_messages([UserMessage(content=parts)])
        assert len(result) == 1
        content = result[0]["content"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What?"}
        assert content[1] == {"type": "image_url", "image_url": {"url": "https://img.png", "detail": "high"}}

    def test_assistant_message_text(self) -> None:
        result = map_messages([AssistantMessage(content="Hi!")])
        assert result == [{"role": "assistant", "content": "Hi!"}]

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments="{}"))
        result = map_messages([AssistantMessage(tool_calls=[tc])])
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert msg["tool_calls"] == [  # pyright: ignore[reportTypedDictNotRequiredAccess]
            {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]

    def test_tool_message(self) -> None:
        result = map_messages([ToolMessage(content="result", tool_call_id="tc1")])
        assert result == [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]

    def test_mixed_messages(self) -> None:
        messages = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        result = map_messages(messages)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["system", "user", "assistant"]


# MARK: map_tools


class TestMapTools:
    def test_minimal_tool(self) -> None:
        tools = [Tool(function=FunctionDefinition(name="f"))]
        result = map_tools(tools)
        assert result == [{"type": "function", "function": {"name": "f"}}]

    def test_full_tool(self) -> None:
        tools = [
            Tool(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                    strict=True,
                )
            )
        ]
        result = map_tools(tools)
        fn = result[0]["function"]
        assert fn["name"] == "get_weather"
        assert fn["description"] == "Get weather"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert fn["parameters"] == {"type": "object"}  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert fn["strict"] is True  # pyright: ignore[reportTypedDictNotRequiredAccess]


# MARK: map_response_format


class TestMapResponseFormat:
    def test_text(self) -> None:
        assert map_response_format(TextResponseFormat()) == {"type": "text"}

    def test_json_object(self) -> None:
        assert map_response_format(JsonObjectResponseFormat()) == {"type": "json_object"}

    def test_json_schema_minimal(self) -> None:
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        result = map_response_format(rf)
        assert result == {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }

    def test_json_schema_full(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="test",
            json_schema={"type": "object"},
            description="A test",
            strict=True,
        )
        result = map_response_format(rf)
        assert result == {
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
        result = map_response_input(items)
        assert result == [
            {"role": "user", "content": "call the tool"},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 42}'},
        ]


# MARK: map_chat_completion


class TestMapChatCompletion:
    def test_basic(self, chat_completion: ChatCompletion, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_chat_completion(chat_completion, "openai", noop_cost_fn)
        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-4o",
            provider="openai",
            finish_reason="stop",
        )

    def test_with_tool_calls(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        tool_calls: list[ChatCompletionMessageToolCallUnion] = [
            ChatCompletionMessageToolCall(
                id="tc1",
                type="function",
                function=ToolCallFunction(name="f", arguments='{"x": 1}'),
            )
        ]
        completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        result = map_chat_completion(completion, "openai", noop_cost_fn)
        assert result.content is None
        assert result.tool_calls == [
            ToolCall(id="tc1", function=FunctionCallResult(name="f", arguments='{"x": 1}')),
        ]

    def test_with_cache_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=50),
            ),
        )
        result = map_chat_completion(completion, "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_none_usage(self, chat_completion: ChatCompletion, noop_cost_fn: Any) -> None:  # noqa: ANN401
        chat_completion.usage = None
        result = map_chat_completion(chat_completion, "openai", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None

    def test_cost_from_calculator(self, chat_completion: ChatCompletion, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_chat_completion(chat_completion, "openai", noop_cost_fn)
        assert result.cost == Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0)

    def test_cost_none_when_unknown(self, chat_completion: ChatCompletion, none_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_chat_completion(chat_completion, "openai", none_cost_fn)
        assert result.cost is None


# MARK: map_chat_chunk


class TestMapChatChunk:
    def test_content_delta(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="Hello"),
                    index=0,
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        )
        result = map_chat_chunk(chunk)
        assert result == ChatChunk(delta="Hello", model="gpt-4o")

    def test_tool_call_delta(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc1",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(name="f", arguments='{"x":'),
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        )
        result = map_chat_chunk(chunk)
        assert result.tool_call_deltas == [
            ToolCallDelta(
                index=0,
                id="tc1",
                type="function",
                function=FunctionCallDelta(name="f", arguments='{"x":'),
            ),
        ]

    def test_tool_call_delta_without_function(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[ChoiceDeltaToolCall(index=0, id="tc1", type="function", function=None)]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        )
        result = map_chat_chunk(chunk)
        assert result.tool_call_deltas == [
            ToolCallDelta(index=0, id="tc1", type="function", function=None),
        ]

    def test_final_chunk_with_usage(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        result = map_chat_chunk(chunk)
        assert result == ChatChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="gpt-4o",
        )

    def test_usage_chunk_with_cache(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
            ),
        )
        result = map_chat_chunk(chunk)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 3

    def test_empty_choices(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        )
        result = map_chat_chunk(chunk)
        assert result == ChatChunk(model="gpt-4o")


# MARK: map_embedding_response


class TestMapEmbeddingResponse:
    def test_single_embedding(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = CreateEmbeddingResponse(
            data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage=EmbUsage(prompt_tokens=5, total_tokens=5),
        )
        result = map_embedding_response(response, "openai", noop_cost_fn)
        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="text-embedding-3-small",
            provider="openai",
        )

    def test_multiple_embeddings_sorted_by_index(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        response = CreateEmbeddingResponse(
            data=[
                Embedding(embedding=[0.3, 0.4], index=1, object="embedding"),
                Embedding(embedding=[0.1, 0.2], index=0, object="embedding"),
            ],
            model="text-embedding-3-small",
            object="list",
            usage=EmbUsage(prompt_tokens=10, total_tokens=10),
        )
        result = map_embedding_response(response, "openai", noop_cost_fn)
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]


# MARK: map_responses_response


class TestMapResponsesResponse:
    def test_basic(self, responses_mock: MagicMock, noop_cost_fn: Any) -> None:  # noqa: ANN401
        result = map_responses_response(responses_mock, "openai", noop_cost_fn)
        assert result == ResponseResponse(
            id="resp_123",
            output_text="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            model="gpt-4o",
            provider="openai",
        )

    def test_with_cached_tokens(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        mock = MagicMock()
        mock.id = "resp_123"
        mock.output_text = "Hello!"
        mock.model = "gpt-4o"
        mock.usage.input_tokens = 10
        mock.usage.output_tokens = 5
        mock.usage.input_tokens_details.cached_tokens = 50
        result = map_responses_response(mock, "openai", noop_cost_fn)
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 50

    def test_none_usage(self, noop_cost_fn: Any) -> None:  # noqa: ANN401
        mock = MagicMock()
        mock.id = "resp_456"
        mock.output_text = "Hi!"
        mock.model = "gpt-4o"
        mock.usage = None
        result = map_responses_response(mock, "openai", noop_cost_fn)
        assert result.usage is None
        assert result.cost is None
