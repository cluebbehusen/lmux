"""Tests for lmux core type definitions."""

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


class TestContentParts:
    def test_text_content(self) -> None:
        tc = TextContent(text="hello")
        assert tc.type == "text"
        assert tc.text == "hello"

    def test_image_content_defaults(self) -> None:
        ic = ImageContent(url="https://example.com/img.png")
        assert ic.type == "image_url"
        assert ic.url == "https://example.com/img.png"
        assert ic.detail == "auto"

    def test_image_content_custom_detail(self) -> None:
        ic = ImageContent(url="https://example.com/img.png", detail="high")
        assert ic.detail == "high"


class TestMessages:
    def test_system_message(self) -> None:
        msg = SystemMessage(content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."

    def test_developer_message(self) -> None:
        msg = DeveloperMessage(content="Be concise.")
        assert msg.role == "developer"
        assert msg.content == "Be concise."

    def test_user_message_text(self) -> None:
        msg = UserMessage(content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_user_message_multimodal(self) -> None:
        parts = [TextContent(text="What's this?"), ImageContent(url="https://example.com/img.png")]
        msg = UserMessage(content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_assistant_message_text(self) -> None:
        msg = AssistantMessage(content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        msg = AssistantMessage(tool_calls=[tc])
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_tool_message(self) -> None:
        msg = ToolMessage(content='{"temp": 72}', tool_call_id="tc_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"


class TestTools:
    def test_function_definition_minimal(self) -> None:
        fd = FunctionDefinition(name="get_weather")
        assert fd.name == "get_weather"
        assert fd.description is None
        assert fd.parameters is None
        assert fd.strict is None

    def test_function_definition_full(self) -> None:
        fd = FunctionDefinition(
            name="get_weather",
            description="Get the weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            strict=True,
        )
        assert fd.description == "Get the weather"
        assert fd.parameters is not None
        assert fd.strict is True

    def test_tool(self) -> None:
        tool = Tool(function=FunctionDefinition(name="f"))
        assert tool.type == "function"
        assert tool.function.name == "f"

    def test_tool_call(self) -> None:
        tc = ToolCall(id="tc_1", function=FunctionCallResult(name="f", arguments="{}"))
        assert tc.type == "function"
        assert tc.id == "tc_1"
        assert tc.function.name == "f"
        assert tc.function.arguments == "{}"

    def test_tool_call_delta_minimal(self) -> None:
        tcd = ToolCallDelta(index=0)
        assert tcd.index == 0
        assert tcd.id is None
        assert tcd.function is None

    def test_tool_call_delta_full(self) -> None:
        tcd = ToolCallDelta(
            index=0,
            id="tc_1",
            type="function",
            function=FunctionCallDelta(name="f", arguments='{"x":'),
        )
        assert tcd.id == "tc_1"
        assert tcd.function is not None
        assert tcd.function.name == "f"
        assert tcd.function.arguments == '{"x":'

    def test_function_call_delta_empty(self) -> None:
        fcd = FunctionCallDelta()
        assert fcd.name is None
        assert fcd.arguments is None


class TestResponseFormat:
    def test_text(self) -> None:
        rf = TextResponseFormat()
        assert rf.type == "text"

    def test_json_object(self) -> None:
        rf = JsonObjectResponseFormat()
        assert rf.type == "json_object"

    def test_json_schema_minimal(self) -> None:
        rf = JsonSchemaResponseFormat(name="person", json_schema={"type": "object"})
        assert rf.type == "json_schema"
        assert rf.name == "person"
        assert rf.description is None
        assert rf.strict is None

    def test_json_schema_full(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="person",
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            description="A person",
            strict=True,
        )
        assert rf.description == "A person"
        assert rf.strict is True


class TestUsageAndCost:
    def test_usage_minimal(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5)
        assert u.input_tokens == 10
        assert u.output_tokens == 5
        assert u.cache_read_tokens is None
        assert u.cache_creation_tokens is None

    def test_usage_with_cache(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3, cache_creation_tokens=2)
        assert u.cache_read_tokens == 3
        assert u.cache_creation_tokens == 2

    def test_cost(self) -> None:
        c = Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03)
        assert c.currency == "USD"
        assert c.cache_read_cost is None
        assert c.cache_creation_cost is None

    def test_cost_with_cache(self) -> None:
        c = Cost(
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.035,
            cache_read_cost=0.003,
            cache_creation_cost=0.002,
        )
        assert c.cache_read_cost == 0.003
        assert c.cache_creation_cost == 0.002


class TestChatResponse:
    def test_basic(self) -> None:
        r = ChatResponse(
            content="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=None,
            model="gpt-4o",
            provider="openai",
        )
        assert r.content == "Hello!"
        assert r.tool_calls is None
        assert r.finish_reason is None
        assert r.cost is None

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", function=FunctionCallResult(name="f", arguments="{}"))
        r = ChatResponse(
            content=None,
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03),
            model="gpt-4o",
            provider="openai",
            finish_reason="tool_calls",
        )
        assert r.content is None
        assert r.tool_calls is not None
        assert len(r.tool_calls) == 1
        assert r.finish_reason == "tool_calls"


class TestChatChunk:
    def test_content_delta(self) -> None:
        c = ChatChunk(delta="Hello")
        assert c.delta == "Hello"
        assert c.tool_call_deltas is None
        assert c.usage is None

    def test_final_chunk_with_usage(self) -> None:
        c = ChatChunk(
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03),
            finish_reason="stop",
            model="gpt-4o",
        )
        assert c.usage is not None
        assert c.cost is not None
        assert c.finish_reason == "stop"

    def test_tool_call_delta_chunk(self) -> None:
        c = ChatChunk(tool_call_deltas=[ToolCallDelta(index=0, id="tc_1", type="function")])
        assert c.tool_call_deltas is not None
        assert len(c.tool_call_deltas) == 1


class TestEmbeddingResponse:
    def test_basic(self) -> None:
        r = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=None,
            model="text-embedding-3-small",
            provider="openai",
        )
        assert len(r.embeddings) == 1
        assert r.embeddings[0] == [0.1, 0.2, 0.3]

    def test_multiple_embeddings(self) -> None:
        r = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            usage=Usage(input_tokens=10, output_tokens=0),
            cost=None,
            model="text-embedding-3-small",
            provider="openai",
        )
        assert len(r.embeddings) == 2


class TestResponseResponse:
    def test_basic(self) -> None:
        r = ResponseResponse(
            id="resp_123",
            output_text="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=None,
            model="gpt-4o",
            provider="openai",
        )
        assert r.id == "resp_123"
        assert r.output_text == "Hello!"


class TestSerialization:
    def test_message_roundtrip(self) -> None:
        msg = UserMessage(content="hello")
        data = msg.model_dump()
        assert data == {"role": "user", "content": "hello"}
        restored = UserMessage.model_validate(data)
        assert restored == msg

    def test_chat_response_roundtrip(self) -> None:
        r = ChatResponse(
            content="Hi",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost=Cost(input_cost=0.001, output_cost=0.002, total_cost=0.003),
            model="gpt-4o",
            provider="openai",
        )
        data = r.model_dump()
        restored = ChatResponse.model_validate(data)
        assert restored == r
