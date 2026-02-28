"""Tests for lmux core type definitions."""

from lmux.types import (
    AssistantMessage,
    ChatResponse,
    Cost,
    FunctionCallResult,
    ImageContent,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)


class TestMultimodalContent:
    def test_user_message_with_parts(self) -> None:
        image = ImageContent(url="https://example.com/img.png", detail="high")
        msg = UserMessage(content=[TextContent(text="What's this?"), image])
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert image.detail == "high"


class TestAssistantToolCalls:
    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'))
        msg = AssistantMessage(tool_calls=[tc])
        assert msg.content is None
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].function.name == "get_weather"


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
