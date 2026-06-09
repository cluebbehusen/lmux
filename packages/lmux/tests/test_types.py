"""Tests for lmux core type definitions."""

import pydantic
import pytest

from lmux.types import (
    AssistantMessage,
    CachePointContent,
    ChatResponse,
    Cost,
    FunctionCallResult,
    ImageContent,
    ServerToolResult,
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

    def test_chat_response_with_server_tool_results(self) -> None:
        r = ChatResponse(
            content="The result is 42.",
            server_tool_results=[
                ServerToolResult(
                    name="code_execution",
                    input={"code": "print(42)", "language": "PYTHON"},
                    output="42\n",
                    provider_specific_fields={"outcome": "OUTCOME_OK"},
                ),
            ],
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=None,
            model="gemini-2.0-flash",
            provider="google",
        )
        data = r.model_dump()
        restored = ChatResponse.model_validate(data)
        assert restored == r


class TestContentPartValidation:
    def test_malformed_part_raises(self) -> None:
        for bad in [{"txt": "hello"}, {}, {"ur1": "https://example.com"}]:
            with pytest.raises(pydantic.ValidationError):
                _ = UserMessage.model_validate({"role": "user", "content": [bad]})

    def test_valid_parts_validate(self) -> None:
        msg = UserMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "url": "https://example.com/i.png"},
                    {"type": "cache_point"},
                ],
            }
        )
        assert isinstance(msg.content, list)
        assert [type(p).__name__ for p in msg.content] == ["TextContent", "ImageContent", "CachePointContent"]

    def test_cache_point_ttl_is_open_string(self) -> None:
        # TTL is provider-validated; lmux passes it through verbatim.
        part = CachePointContent(ttl="30m")
        assert part.ttl == "30m"
        assert CachePointContent().ttl is None
