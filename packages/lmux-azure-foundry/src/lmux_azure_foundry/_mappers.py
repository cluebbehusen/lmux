"""Internal mappers between lmux types and Azure AI Foundry (OpenAI-compatible) JSON.

Input mappers emit plain JSON dicts for the request body; output mappers consume
the raw JSON dicts returned by the REST API (not SDK objects).
"""

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
    EmbeddingResponse,
    FunctionCallDelta,
    FunctionCallResult,
    ImageContent,
    JsonObjectResponseFormat,
    Message,
    ResponseFormat,
    ResponseInputItem,
    ResponseResponse,
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

type CostCalculator = Callable[[str, Usage], Cost | None]
type Json = dict[str, Any]


# MARK: Input Mappers (lmux -> OpenAI-compatible JSON)


def map_messages(messages: Sequence[Message]) -> list[Json]:
    """Convert lmux Messages to OpenAI-compatible message dicts."""
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
    """Convert lmux Tools to OpenAI tool param dicts."""
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
    """Convert lmux ToolChoice to OpenAI tool_choice param."""
    if isinstance(tc, ToolChoiceFunction):
        return {"type": "function", "function": {"name": tc.name}}
    return tc  # "auto", "required", "none"


def map_response_format(rf: ResponseFormat) -> Json:
    """Convert lmux ResponseFormat to OpenAI response_format param dict."""
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


def map_response_input(input: str | Sequence[ResponseInputItem]) -> str | list[Json]:  # noqa: A002
    """Convert lmux ResponseInputItem sequence to OpenAI-compatible dicts."""
    if isinstance(input, str):
        return input
    return [item.model_dump(exclude_none=True) for item in input]


# MARK: Output Mappers (OpenAI-compatible JSON -> lmux)


def _map_function_tool_call(tc: Json) -> ToolCall:
    fn = tc["function"]
    return ToolCall(id=tc["id"], function=FunctionCallResult(name=fn["name"], arguments=fn["arguments"]))


def map_chat_completion(completion: Json, provider_name: str, cost_fn: CostCalculator) -> ChatResponse:
    """Convert an OpenAI-compatible chat completion JSON body to an lmux ChatResponse."""
    choice = completion["choices"][0]
    message = choice["message"]

    tool_calls: list[ToolCall] | None = None
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        tool_calls = [_map_function_tool_call(tc) for tc in raw_tool_calls if tc.get("type") == "function"]

    usage = _map_usage(completion.get("usage"))
    model = completion["model"]
    cost = cost_fn(model, usage) if usage else None

    return ChatResponse(
        content=message.get("content"),
        reasoning=message.get("reasoning_content"),
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
        finish_reason=choice.get("finish_reason"),
    )


def _map_usage(usage: Json | None) -> Usage | None:
    """Extract Usage from a completion/chunk usage dict, or None if absent."""
    if usage is None:
        return None
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        cache_read_tokens=prompt_details.get("cached_tokens") or None,
        reasoning_tokens=completion_details.get("reasoning_tokens") or None,
    )


def map_chat_chunk(chunk: Json, provider_name: str) -> ChatChunk:
    """Convert an OpenAI-compatible streaming chunk JSON to an lmux ChatChunk."""
    delta_text: str | None = None
    reasoning_delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    finish_reason: str | None = None

    choices = chunk.get("choices")
    if choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        delta_text = delta.get("content")
        reasoning_delta = delta.get("reasoning_content")
        finish_reason = choice.get("finish_reason")
        raw_tool_calls = delta.get("tool_calls")
        if raw_tool_calls:
            tool_call_deltas = [_map_tool_call_delta(tc) for tc in raw_tool_calls]

    return ChatChunk(
        delta=delta_text,
        reasoning_delta=reasoning_delta,
        tool_call_deltas=tool_call_deltas,
        usage=_map_usage(chunk.get("usage")),
        finish_reason=finish_reason,
        model=chunk.get("model"),
        provider=provider_name,
    )


def _map_tool_call_delta(tc: Json) -> ToolCallDelta:
    fn = tc.get("function")
    return ToolCallDelta(
        index=tc["index"],
        id=tc.get("id"),
        type="function" if tc.get("type") == "function" else None,
        function=FunctionCallDelta(name=fn.get("name"), arguments=fn.get("arguments")) if fn else None,
    )


def map_embedding_response(response: Json, provider_name: str, cost_fn: CostCalculator) -> EmbeddingResponse:
    """Convert an OpenAI-compatible embeddings JSON body to an lmux EmbeddingResponse."""
    data = sorted(response["data"], key=lambda item: item["index"])
    embeddings = [item["embedding"] for item in data]
    usage = Usage(input_tokens=response["usage"]["prompt_tokens"], output_tokens=0)
    cost = cost_fn(response["model"], usage)
    return EmbeddingResponse(
        embeddings=embeddings,
        usage=usage,
        cost=cost,
        model=response["model"],
        provider=provider_name,
    )


def map_responses_response(response: Json, provider_name: str, cost_fn: CostCalculator) -> ResponseResponse:
    """Convert an OpenAI-compatible Responses API JSON body to an lmux ResponseResponse."""
    usage = _map_responses_usage(response.get("usage"))
    model = response["model"]
    cost = cost_fn(model, usage) if usage else None
    return ResponseResponse(
        id=response["id"],
        output_text=_extract_output_text(response.get("output") or []),
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
    )


def _map_responses_usage(usage: Json | None) -> Usage | None:
    if usage is None:
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return Usage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_read_tokens=input_details.get("cached_tokens") or None,
        reasoning_tokens=output_details.get("reasoning_tokens") or None,
    )


def _extract_output_text(output: list[Json]) -> str:
    """Aggregate all ``output_text`` blocks from the Responses API ``output`` list."""
    return "".join(
        content.get("text", "")
        for item in output
        if item.get("type") == "message"
        for content in item.get("content") or []
        if content.get("type") == "output_text"
    )
