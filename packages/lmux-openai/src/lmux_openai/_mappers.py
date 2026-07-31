"""Internal mappers between lmux types and OpenAI (REST) JSON."""

import re
from collections.abc import Callable, Sequence
from typing import Any

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
    ResponseInputMessage,
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
from lmux_openai._wire import (
    WireChunk,
    WireCompletion,
    WireCompletionUsage,
    WireEmbeddingResponse,
    WireResponsesResponse,
    WireResponsesUsage,
    WireToolCallDelta,
)

type CostCalculator = Callable[[str, Usage], Cost | None]
type Json = dict[str, Any]

_GPT_MODEL_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)")


# MARK: Input Mappers (lmux -> OpenAI JSON)


def supports_explicit_prompt_cache(model: str) -> bool:
    """Return whether the model is in the GPT-5.6-or-later explicit-cache generation."""
    match = _GPT_MODEL_RE.match(model)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (5, 6)


def map_messages(messages: Sequence[Message], *, explicit_cache: bool = False) -> list[Json]:
    """Convert lmux Messages to OpenAI-compatible message dicts.

    When ``explicit_cache`` is set (gpt-5.6+), a ``CachePointContent`` becomes a
    ``prompt_cache_breakpoint`` on the preceding content block; otherwise cache points are dropped
    (older models have no explicit representation). ``CachePointContent.ttl`` is per-breakpoint, but
    OpenAI's cache lifetime is a request-wide option (``prompt_cache_options.ttl``, currently only
    ``"30m"``, the default), so the per-breakpoint ``ttl`` is not mapped and has no effect.
    """
    result: list[Json] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, DeveloperMessage):
            result.append({"role": "developer", "content": msg.content})
        elif isinstance(msg, UserMessage):
            content, leading_cache_point = _map_user_content(msg.content, explicit_cache=explicit_cache)
            if leading_cache_point is not None and result:
                # Nothing precedes the marker in this message — cache the prior message's prefix.
                _attach_breakpoint(result[-1])
            if isinstance(content, str) or content or not msg.content:
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


def has_cache_breakpoint(messages: list[Json]) -> bool:
    """True if any mapped message carries a prompt_cache_breakpoint, i.e. the request opts into explicit mode."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any("prompt_cache_breakpoint" in block for block in content):
            return True
    return False


def _map_tool_call_param(tc: ToolCall) -> Json:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _map_user_content(
    content: str | list[ContentPart], *, explicit_cache: bool
) -> tuple[str | list[Json], CachePointContent | None]:
    """Map user content, translating cache points to breakpoints on the preceding block.

    Returns ``(mapped, leading_cache_point)`` — a cache point with no preceding block in this message
    is returned so the caller can attach it to whatever precedes the message.
    """
    return _map_content_with_cache_points(content, explicit_cache=explicit_cache, mapper=_map_content_part)


def _map_content_with_cache_points(
    content: str | list[ContentPart],
    *,
    explicit_cache: bool,
    mapper: Callable[[TextContent | ImageContent], Json],
) -> tuple[str | list[Json], CachePointContent | None]:
    if isinstance(content, str):
        return content, None
    blocks: list[Json] = []
    leading_cache_point: CachePointContent | None = None
    for part in content:
        if isinstance(part, CachePointContent):
            if not explicit_cache:
                continue
            if not blocks:
                if leading_cache_point is None:
                    leading_cache_point = part
            elif "prompt_cache_breakpoint" not in blocks[-1]:
                blocks[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        else:
            blocks.append(mapper(part))
    return blocks, leading_cache_point


def _attach_breakpoint(message: Json, *, text_type: str = "text") -> None:
    """Attach a prompt_cache_breakpoint to the last content block of ``message`` (the first marker wins)."""
    content = message.get("content")
    if content is None:
        return
    if isinstance(content, str):
        if not content:
            return
        message["content"] = [{"type": text_type, "text": content, "prompt_cache_breakpoint": {"mode": "explicit"}}]
        return
    blocks: list[Json] = content
    if blocks and "prompt_cache_breakpoint" not in blocks[-1]:
        blocks[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}


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
    schema_dict: Json = {"name": rf.name, "schema": rf.json_schema}
    if rf.description is not None:
        schema_dict["description"] = rf.description
    if rf.strict is not None:
        schema_dict["strict"] = rf.strict
    return {"type": "json_schema", "json_schema": schema_dict}


def map_response_input(
    input: str | Sequence[ResponseInputItem],  # noqa: A002
    *,
    explicit_cache: bool = False,
) -> str | list[Json]:
    """Convert lmux ResponseInputItem sequence to OpenAI-compatible dicts."""
    if isinstance(input, str):
        return input
    result: list[Json] = []
    for item in input:
        if not isinstance(item, ResponseInputMessage):
            result.append(item.model_dump(exclude_none=True))
            continue
        content, leading_cache_point = _map_content_with_cache_points(
            item.content, explicit_cache=explicit_cache, mapper=_map_response_content_part
        )
        if leading_cache_point is not None and result:
            _attach_breakpoint(result[-1], text_type="input_text")
        if isinstance(content, str) or content or not item.content:
            result.append({"role": item.role, "content": content})
    return result


def _map_response_content_part(part: TextContent | ImageContent) -> Json:
    if isinstance(part, TextContent):
        return {"type": "input_text", "text": part.text}
    return {"type": "input_image", "image_url": part.url, "detail": part.detail}


# MARK: Output Mappers (OpenAI wire models -> lmux)


def map_chat_completion(completion: WireCompletion, provider_name: str, cost_fn: CostCalculator) -> ChatResponse:
    """Convert a validated OpenAI chat completion to an lmux ChatResponse."""
    choice = completion.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        # A ``function`` payload is present only for function tool calls; other types (e.g. custom) are dropped.
        tool_calls = [
            ToolCall(id=tc.id, function=FunctionCallResult(name=tc.function.name, arguments=tc.function.arguments))
            for tc in message.tool_calls
            if tc.function is not None
        ]

    usage = _map_completion_usage(completion.usage)
    cost = cost_fn(completion.model, usage) if usage else None

    return ChatResponse(
        content=message.content,
        reasoning=message.reasoning_content,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=completion.model,
        provider=provider_name,
        finish_reason=choice.finish_reason,
    )


def _map_completion_usage(usage: WireCompletionUsage | None) -> Usage | None:
    """Extract Usage from a completion/chunk usage model, or None if absent."""
    if usage is None:
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    # gpt-5.6+ bills cache writes; older models report no cache_write_tokens (writes free).
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=(prompt_details.cached_tokens if prompt_details else None) or None,
        cache_creation_tokens=(prompt_details.cache_write_tokens if prompt_details else None) or None,
        reasoning_tokens=(completion_details.reasoning_tokens if completion_details else None) or None,
    )


def map_chat_chunk(chunk: WireChunk, provider_name: str) -> ChatChunk:
    """Convert a validated OpenAI streaming chunk to an lmux ChatChunk."""
    delta_text: str | None = None
    reasoning_delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    finish_reason: str | None = None

    if chunk.choices:
        choice = chunk.choices[0]
        delta = choice.delta
        delta_text = delta.content
        reasoning_delta = delta.reasoning_content
        finish_reason = choice.finish_reason
        if delta.tool_calls:
            tool_call_deltas = [_map_tool_call_delta(tc) for tc in delta.tool_calls]

    return ChatChunk(
        delta=delta_text,
        reasoning_delta=reasoning_delta,
        tool_call_deltas=tool_call_deltas,
        usage=_map_completion_usage(chunk.usage),
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


def map_embedding_response(
    response: WireEmbeddingResponse, provider_name: str, cost_fn: CostCalculator
) -> EmbeddingResponse:
    """Convert a validated OpenAI embeddings response to an lmux EmbeddingResponse."""
    embeddings = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    usage = Usage(input_tokens=response.usage.prompt_tokens, output_tokens=0)
    cost = cost_fn(response.model, usage)
    return EmbeddingResponse(
        embeddings=embeddings,
        usage=usage,
        cost=cost,
        model=response.model,
        provider=provider_name,
    )


def map_responses_response(
    response: WireResponsesResponse, provider_name: str, cost_fn: CostCalculator
) -> ResponseResponse:
    """Convert a validated OpenAI Responses API response to an lmux ResponseResponse."""
    usage = _map_responses_usage(response.usage)
    cost = cost_fn(response.model, usage) if usage else None
    return ResponseResponse(
        id=response.id,
        output_text=_extract_output_text(response),
        usage=usage,
        cost=cost,
        model=response.model,
        provider=provider_name,
    )


def _extract_output_text(response: WireResponsesResponse) -> str:
    """Concatenate output_text content parts across all message output items."""
    return "".join(
        content.text or ""
        for item in response.output or []
        if item.type == "message"
        for content in item.content or []
        if content.type == "output_text"
    )


def _map_responses_usage(usage: WireResponsesUsage | None) -> Usage | None:
    """Extract Usage from a Responses API usage model, or None if absent."""
    if usage is None:
        return None
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=(input_details.cached_tokens if input_details else None) or None,
        cache_creation_tokens=(input_details.cache_write_tokens if input_details else None) or None,
        reasoning_tokens=(output_details.reasoning_tokens if output_details else None) or None,
    )
