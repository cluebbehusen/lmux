"""Internal mappers between lmux types and Groq (OpenAI-compatible) JSON."""

import copy
from collections.abc import Callable, Sequence
from typing import Any

from lmux.schema import add_additional_properties_false
from lmux.types import (
    AssistantMessage,
    CachePointContent,
    ChatChunk,
    ChatResponse,
    ContentPart,
    Cost,
    DeveloperMessage,
    FunctionCallDelta,
    FunctionCallResult,
    ImageContent,
    JsonObjectResponseFormat,
    Message,
    ResponseFormat,
    SystemMessage,
    TextContent,
    TextResponseFormat,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolChoiceFunction,
    Usage,
    UserMessage,
)
from lmux_groq._wire import (
    WireChunk,
    WireCompletion,
    WireToolCall,
    WireToolCallDelta,
    WireUsage,
)

type CostCalculator = Callable[[str, Usage], Cost | None]
type Json = dict[str, Any]


# MARK: Input Mappers (lmux -> Groq JSON)


def map_messages(messages: Sequence[Message]) -> list[Json]:
    """Convert lmux Messages to Groq-compatible message dicts."""
    result: list[Json] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, DeveloperMessage):
            result.append({"role": "developer", "content": msg.content})
        elif isinstance(msg, UserMessage):
            content = _map_user_content(msg.content)
            if isinstance(content, list) and not content and msg.content:
                continue  # message held only cache points, which this provider has no representation for
            result.append({"role": "user", "content": content})
        elif isinstance(msg, AssistantMessage):
            d: Json = {"role": "assistant"}
            if msg.content is not None:
                d["content"] = msg.content
            if msg.tool_calls:
                d["tool_calls"] = [_map_tool_call_param(tc) for tc in msg.tool_calls]
            result.append(d)
        else:
            result.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
    return result


def _map_tool_call_param(tc: ToolCall) -> Json:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _map_user_content(content: str | list[ContentPart]) -> str | list[Json]:
    if isinstance(content, str):
        return content
    # Cache points are dropped: this provider caches implicitly and has no explicit representation.
    return [_map_content_part(part) for part in content if not isinstance(part, CachePointContent)]


def _map_content_part(part: TextContent | ImageContent) -> Json:
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.text}
    return {"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}


def map_tools(tools: list[Tool]) -> list[Json]:
    """Convert lmux Tools to Groq tool param dicts."""
    result: list[Json] = []
    for tool in tools:
        fn: Json = {"name": tool.function.name}
        if tool.function.description is not None:
            fn["description"] = tool.function.description
        if tool.function.parameters is not None:
            fn["parameters"] = tool.function.parameters
        if tool.function.strict is not None:
            fn["strict"] = tool.function.strict
        result.append({"type": "function", "function": fn})
    return result


def map_tool_choice(tc: ToolChoice) -> str | Json:
    """Convert lmux ToolChoice to Groq tool_choice param."""
    if isinstance(tc, ToolChoiceFunction):
        return {"type": "function", "function": {"name": tc.name}}
    return tc  # "auto", "required", "none"


def map_response_format(rf: ResponseFormat) -> Json:
    """Convert lmux ResponseFormat to Groq response_format param dict."""
    if isinstance(rf, TextResponseFormat):
        return {"type": "text"}
    if isinstance(rf, JsonObjectResponseFormat):
        return {"type": "json_object"}
    patched = copy.deepcopy(rf.json_schema)
    add_additional_properties_false(patched)
    schema_dict: Json = {"name": rf.name, "schema": patched}
    if rf.description is not None:
        schema_dict["description"] = rf.description
    if rf.strict is not None:
        schema_dict["strict"] = rf.strict
    return {"type": "json_schema", "json_schema": schema_dict}


# MARK: Output Mappers (Groq wire models -> lmux)


def _map_function_tool_call(tc: WireToolCall) -> ToolCall:
    return ToolCall(id=tc.id, function=FunctionCallResult(name=tc.function.name, arguments=tc.function.arguments))


def map_chat_completion(completion: WireCompletion, provider_name: str, cost_fn: CostCalculator) -> ChatResponse:
    """Convert a validated Groq chat completion to an lmux ChatResponse."""
    choice = completion.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        tool_calls = [_map_function_tool_call(tc) for tc in message.tool_calls if tc.type == "function"]

    usage = _map_usage(completion.usage)
    cost = cost_fn(completion.model, usage) if usage else None

    return ChatResponse(
        content=message.content,
        reasoning=message.reasoning,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=completion.model,
        provider=provider_name,
        finish_reason=choice.finish_reason,
    )


def _map_usage(usage: WireUsage | None) -> Usage | None:
    """Extract Usage from a completion/chunk usage model, or None if absent."""
    if usage is None:
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=(prompt_details.cached_tokens if prompt_details else None) or None,
        reasoning_tokens=(completion_details.reasoning_tokens if completion_details else None) or None,
    )


def map_chat_chunk(chunk: WireChunk, provider_name: str) -> ChatChunk:
    """Convert a validated Groq streaming chunk to an lmux ChatChunk."""
    delta_text: str | None = None
    reasoning_delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    finish_reason: str | None = None

    if chunk.choices:
        choice = chunk.choices[0]
        delta = choice.delta
        delta_text = delta.content
        reasoning_delta = delta.reasoning
        finish_reason = choice.finish_reason
        if delta.tool_calls:
            tool_call_deltas = [_map_tool_call_delta(tc) for tc in delta.tool_calls]

    return ChatChunk(
        delta=delta_text,
        reasoning_delta=reasoning_delta,
        tool_call_deltas=tool_call_deltas,
        usage=_map_usage(chunk.usage),
        finish_reason=finish_reason,
        model=chunk.model,
        provider=provider_name,
    )


def _map_tool_call_delta(tc: WireToolCallDelta) -> ToolCallDelta:
    fn = tc.function
    return ToolCallDelta(
        index=tc.index,
        id=tc.id,
        type="function" if tc.type == "function" else None,
        function=FunctionCallDelta(name=fn.name, arguments=fn.arguments) if fn else None,
    )
