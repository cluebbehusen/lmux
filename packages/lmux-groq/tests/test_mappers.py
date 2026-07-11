"""Tests for Groq JSON mappers."""

from typing import Any

from lmux.types import (
    AssistantMessage,
    CachePointContent,
    Cost,
    DeveloperMessage,
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
    ToolChoiceFunction,
    ToolMessage,
    Usage,
    UserMessage,
)
from lmux_groq._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
)
from lmux_groq._wire import WireChunk, WireCompletion


def _noop_cost(model: str, usage: Usage) -> Cost | None:  # noqa: ARG001
    return None


class TestMapMessages:
    def test_system(self) -> None:
        assert map_messages([SystemMessage(content="s")]) == [{"role": "system", "content": "s"}]

    def test_developer(self) -> None:
        assert map_messages([DeveloperMessage(content="d")]) == [{"role": "developer", "content": "d"}]

    def test_user_text(self) -> None:
        assert map_messages([UserMessage(content="hi")]) == [{"role": "user", "content": "hi"}]

    def test_user_content_parts(self) -> None:
        result = map_messages([UserMessage(content=[TextContent(text="t"), ImageContent(url="http://x")])])
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "t"},
                    {"type": "image_url", "image_url": {"url": "http://x", "detail": "auto"}},
                ],
            }
        ]

    def test_user_only_cache_points_skipped(self) -> None:
        assert map_messages([UserMessage(content=[CachePointContent()])]) == []

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


class TestMapToolChoice:
    def test_string(self) -> None:
        assert map_tool_choice("auto") == "auto"

    def test_function(self) -> None:
        assert map_tool_choice(ToolChoiceFunction(name="f")) == {"type": "function", "function": {"name": "f"}}


class TestMapResponseFormat:
    def test_text(self) -> None:
        assert map_response_format(TextResponseFormat()) == {"type": "text"}

    def test_json_object(self) -> None:
        assert map_response_format(JsonObjectResponseFormat()) == {"type": "json_object"}

    def test_json_schema_full(self) -> None:
        rf = JsonSchemaResponseFormat(
            name="S", json_schema={"type": "object", "properties": {}}, description="d", strict=True
        )
        out = map_response_format(rf)
        assert out["type"] == "json_schema"
        assert out["json_schema"]["name"] == "S"
        assert out["json_schema"]["description"] == "d"
        assert out["json_schema"]["strict"] is True
        assert out["json_schema"]["schema"]["additionalProperties"] is False

    def test_json_schema_minimal(self) -> None:
        rf = JsonSchemaResponseFormat(name="S", json_schema={"type": "object", "properties": {}})
        out = map_response_format(rf)
        assert "description" not in out["json_schema"]
        assert "strict" not in out["json_schema"]


class TestMapChatCompletion:
    def test_basic(self) -> None:
        completion: dict[str, Any] = {
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        r = map_chat_completion(WireCompletion.model_validate(completion), "groq", _noop_cost)
        assert r.content == "Hi"
        assert r.finish_reason == "stop"
        assert r.usage is not None
        assert r.usage.input_tokens == 10

    def test_tool_calls(self) -> None:
        completion: dict[str, Any] = {
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        r = map_chat_completion(WireCompletion.model_validate(completion), "groq", _noop_cost)
        assert r.tool_calls is not None
        assert r.tool_calls[0].id == "c1"

    def test_reasoning_and_cache(self) -> None:
        completion: dict[str, Any] = {
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "x", "reasoning": "because"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
        r = map_chat_completion(WireCompletion.model_validate(completion), "groq", _noop_cost)
        assert r.reasoning == "because"
        assert r.usage is not None
        assert r.usage.cache_read_tokens == 3
        assert r.usage.reasoning_tokens == 2

    def test_no_usage(self) -> None:
        completion: dict[str, Any] = {
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
        }
        r = map_chat_completion(WireCompletion.model_validate(completion), "groq", _noop_cost)
        assert r.usage is None
        assert r.cost is None


class TestMapChatChunk:
    def test_content_delta(self) -> None:
        chunk: dict[str, Any] = {
            "model": "m",
            "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}],
        }
        assert map_chat_chunk(WireChunk.model_validate(chunk), "groq").delta == "Hel"

    def test_reasoning_delta(self) -> None:
        chunk: dict[str, Any] = {"model": "m", "choices": [{"index": 0, "delta": {"reasoning": "r"}}]}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "groq").reasoning_delta == "r"

    def test_tool_call_delta_with_function(self) -> None:
        chunk: dict[str, Any] = {
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                        ]
                    },
                }
            ],
        }
        c = map_chat_chunk(WireChunk.model_validate(chunk), "groq")
        assert c.tool_call_deltas is not None
        assert c.tool_call_deltas[0].function is not None
        assert c.tool_call_deltas[0].function.name == "f"

    def test_tool_call_delta_without_function(self) -> None:
        chunk: dict[str, Any] = {"model": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}}]}
        c = map_chat_chunk(WireChunk.model_validate(chunk), "groq")
        assert c.tool_call_deltas is not None
        assert c.tool_call_deltas[0].function is None

    def test_finish_and_usage(self) -> None:
        chunk: dict[str, Any] = {
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        c = map_chat_chunk(WireChunk.model_validate(chunk), "groq")
        assert c.finish_reason == "stop"
        assert c.usage is not None
        assert c.usage.input_tokens == 10

    def test_no_choices(self) -> None:
        chunk: dict[str, Any] = {"model": "m", "choices": []}
        assert map_chat_chunk(WireChunk.model_validate(chunk), "groq").delta is None
