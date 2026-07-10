"""Internal mappers between lmux types and Anthropic SDK types."""

import copy
import json
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, cast

from lmux.exceptions import UnsupportedFeatureError
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
    ToolMessage,
    Usage,
    UserMessage,
)

if TYPE_CHECKING:
    from anthropic.types import (
        CacheControlEphemeralParam,
        ImageBlockParam,
        JSONOutputFormatParam,
        MessageDeltaUsage,
        MessageParam,
        OutputConfigParam,
        TextBlockParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
    )
    from anthropic.types import (
        Message as AnthropicMessage,
    )
    from anthropic.types import (
        Usage as AnthropicUsage,
    )
    from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
    from anthropic.types.raw_content_block_start_event import RawContentBlockStartEvent
    from anthropic.types.raw_message_delta_event import RawMessageDeltaEvent
    from anthropic.types.raw_message_start_event import RawMessageStartEvent

type CostCalculator = Callable[[str, Usage], Cost | None]

_DATA_URI_PATTERN = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)

_STOP_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "model_context_window_exceeded": "length",
    "pause_turn": "pause_turn",
}


def _map_stop_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return _STOP_REASON_MAP.get(stop_reason, stop_reason)


# MARK: Input Mappers (lmux -> Anthropic SDK params)


def map_messages(messages: Sequence[Message]) -> tuple[str | list["TextBlockParam"] | None, list["MessageParam"]]:
    """Convert lmux Messages to Anthropic format.

    Returns ``(system, messages_list)`` where ``system`` is extracted from any
    ``SystemMessage`` / ``DeveloperMessage`` instances and the list contains
    only ``user`` and ``assistant`` role messages. ``system`` is a plain string
    unless a cache point applies to it, in which case it is a text block list
    carrying ``cache_control``.
    """
    system_parts: list[str] = []
    system_cache_split: tuple[int, CachePointContent] | None = None
    result: list[MessageParam] = []

    for msg in messages:
        if isinstance(msg, (SystemMessage, DeveloperMessage)):
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            content, leading_cache_point = _map_user_content(msg.content)
            if leading_cache_point is not None:
                if result:
                    _attach_cache_point(result[-1], leading_cache_point)
                elif system_parts and system_cache_split is None:
                    # Cache only the system text seen so far; the first marker wins.
                    system_cache_split = (len(system_parts), leading_cache_point)
                # else: nothing precedes the marker — an empty prefix, dropped
            if isinstance(content, str) or content or not msg.content:
                result.append({"role": "user", "content": content})
        elif isinstance(msg, AssistantMessage):
            result.append(_map_assistant_message(msg))
        else:
            _append_tool_result(result, msg)

    system = _map_system(system_parts, system_cache_split)
    return system, result


def _map_system(
    system_parts: list[str], cache_split: "tuple[int, CachePointContent] | None"
) -> str | list["TextBlockParam"] | None:
    if not system_parts:
        return None
    if cache_split is None:
        return "\n".join(system_parts)
    # Split at the marker position so system text appearing after the cache
    # point stays outside the cached prefix.
    split, cache_point = cache_split
    blocks: list[TextBlockParam] = [
        {"type": "text", "text": "\n".join(system_parts[:split]), "cache_control": _map_cache_control(cache_point)}
    ]
    if len(system_parts) > split:
        blocks.append({"type": "text", "text": "\n".join(system_parts[split:])})
    return blocks


def _map_user_content(
    content: str | list[ContentPart],
) -> tuple[str | list["TextBlockParam | ImageBlockParam"], CachePointContent | None]:
    """Map user content, attaching cache points to the preceding block.

    Returns ``(blocks, leading_cache_point)`` — a cache point with no
    preceding block in the same message is returned for the caller to attach
    to whatever precedes the message (the prior message, or the system prompt).
    """
    if isinstance(content, str):
        return content, None
    blocks: list[TextBlockParam | ImageBlockParam] = []
    leading_cache_point: CachePointContent | None = None
    for part in content:
        if isinstance(part, CachePointContent):
            if not blocks:
                if leading_cache_point is None:
                    leading_cache_point = part
            elif "cache_control" not in blocks[-1]:
                blocks[-1]["cache_control"] = _map_cache_control(part)
        else:
            blocks.append(_map_content_part(part))
    return blocks, leading_cache_point


def _attach_cache_point(message: "MessageParam", cache_point: CachePointContent) -> None:
    """Attach ``cache_control`` to the last content block of ``message``.

    No-ops when the message has nothing cacheable (empty content) or the
    target block already carries a marker (the first cache point wins).
    """
    content = message["content"]
    if isinstance(content, str):
        if not content:
            return
        text_block: TextBlockParam = {"type": "text", "text": content, "cache_control": _map_cache_control(cache_point)}
        message["content"] = [text_block]
        return
    blocks = cast("list[dict[str, object]]", content)
    if blocks and "cache_control" not in blocks[-1]:
        blocks[-1]["cache_control"] = _map_cache_control(cache_point)


def _map_cache_control(cache_point: CachePointContent) -> "CacheControlEphemeralParam":
    cache_control: CacheControlEphemeralParam = {"type": "ephemeral"}
    if cache_point.ttl is not None:
        # ttl is a deliberate passthrough — the API validates the value
        cache_control["ttl"] = cast('Literal["5m", "1h"]', cache_point.ttl)
    return cache_control


def _map_content_part(part: TextContent | ImageContent) -> "TextBlockParam | ImageBlockParam":
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.text}
    return _map_image_content(part)


def _map_image_content(img: ImageContent) -> "ImageBlockParam":
    match = _DATA_URI_PATTERN.match(img.url)
    if match:
        return {"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}}  # pyright: ignore[reportReturnType]  # media_type is a dynamic str, not a literal
    return {"type": "image", "source": {"type": "url", "url": img.url}}


def _map_assistant_message(msg: AssistantMessage) -> "MessageParam":
    content: list[TextBlockParam | ToolUseBlockParam] = []
    if msg.content is not None:
        content.append({"type": "text", "text": msg.content})
    if msg.tool_calls:
        content.extend(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        )
    return {"role": "assistant", "content": content}


def _append_tool_result(result: list["MessageParam"], msg: ToolMessage) -> None:
    """Append a tool_result block, merging consecutive tool results into one user message."""
    tool_block: ToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": msg.content,
    }
    if result and result[-1].get("role") == "user":
        last_content = result[-1].get("content")
        if isinstance(last_content, list) and last_content:
            first = last_content[0]
            if isinstance(first, dict) and first.get("type") == "tool_result":
                last_content.append(tool_block)
                return
    result.append({"role": "user", "content": [tool_block]})


def map_tools(tools: list[Tool]) -> list["ToolParam"]:
    """Convert lmux Tools to Anthropic tool param dicts."""
    result: list[ToolParam] = []
    for tool in tools:
        t: ToolParam = {
            "name": tool.function.name,
            "input_schema": tool.function.parameters or {"type": "object"},
        }
        if tool.function.description is not None:
            t["description"] = tool.function.description
        result.append(t)
    return result


def map_tool_choice(tc: ToolChoice) -> dict[str, str]:
    """Convert lmux ToolChoice to Anthropic tool_choice param."""
    if tc == "none":
        return {"type": "none"}
    if tc == "required":
        return {"type": "any"}
    if isinstance(tc, ToolChoiceFunction):
        return {"type": "tool", "name": tc.name}
    return {"type": "auto"}


_ADAPTIVE_MIN = (4, 6)
_MODEL_GEN_RE = re.compile(r"claude-(?:opus|sonnet|haiku|fable)-(\d+)(?:-(\d{1,2})(?=$|[-:@]))?")


def model_uses_adaptive_thinking(model: str) -> bool:
    """Return True for Claude generations >= 4.6 (adaptive thinking + effort).

    Returns False for <= 4.5 (legacy manual ``budget_tokens`` thinking) and for
    any unparseable string (the legacy path is the safe default: pre-4.6 models
    accept ``budget_tokens``, newer ones require adaptive).
    """
    match = _MODEL_GEN_RE.search(model)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) is not None else 0
    return (major, minor) >= _ADAPTIVE_MIN


def map_response_format(rf: ResponseFormat) -> "OutputConfigParam | None":
    """Convert lmux ResponseFormat to Anthropic output_config dict, or None for text."""
    if isinstance(rf, TextResponseFormat):
        return None
    if isinstance(rf, JsonObjectResponseFormat):
        msg = "JsonObjectResponseFormat is not supported by Anthropic; use JsonSchemaResponseFormat instead"
        raise UnsupportedFeatureError(msg, provider="anthropic")
    patched = copy.deepcopy(rf.json_schema)
    add_additional_properties_false(patched)
    schema_dict: JSONOutputFormatParam = {"type": "json_schema", "schema": patched}
    return {"format": schema_dict}


# MARK: Output Mappers (Anthropic SDK responses -> lmux)


def map_message_response(
    message: "AnthropicMessage",
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert Anthropic Message to lmux ChatResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in message.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=FunctionCallResult(name=block.name, arguments=json.dumps(block.input)),
                )
            )

    content = "\n".join(text_parts) if text_parts else None
    reasoning = "\n".join(thinking_parts) if thinking_parts else None
    usage = _map_usage(message.usage)
    cost = cost_fn(message.model, usage)

    return ChatResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=message.model,
        provider=provider_name,
        finish_reason=_map_stop_reason(message.stop_reason),
    )


def _map_usage(usage: "AnthropicUsage") -> Usage:
    cache_read: int = getattr(usage, "cache_read_input_tokens", None) or 0
    cache_creation: int = getattr(usage, "cache_creation_input_tokens", None) or 0
    return Usage(
        # Anthropic reports input_tokens exclusive of cached tokens; lmux's
        # Usage convention is the inclusive total (see lmux.types.Usage).
        input_tokens=usage.input_tokens + cache_read + cache_creation,
        output_tokens=usage.output_tokens,
        cache_read_tokens=cache_read or None,
        cache_creation_tokens=cache_creation or None,
        cache_creation_tokens_by_ttl=_map_cache_creation_breakdown(usage),
    )


def _map_cache_creation_breakdown(usage: "AnthropicUsage") -> dict[str, int] | None:
    """Map the per-TTL cache-write breakdown (``usage.cache_creation``) if reported."""
    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is None:
        return None
    breakdown: dict[str, int] = {}
    five_minute = getattr(cache_creation, "ephemeral_5m_input_tokens", None)
    one_hour = getattr(cache_creation, "ephemeral_1h_input_tokens", None)
    if five_minute:
        breakdown["5m"] = five_minute
    if one_hour:
        breakdown["1h"] = one_hour
    return breakdown or None


# MARK: Streaming Mappers


def map_message_start(event: "RawMessageStartEvent") -> Usage:
    """Extract input token usage from the message_start event."""
    return _map_usage(event.message.usage)


def map_content_block_start(event: "RawContentBlockStartEvent") -> ChatChunk | None:
    """Map a content_block_start event. Returns a chunk for tool_use blocks only."""
    block = event.content_block
    if block.type == "tool_use":
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=event.index,
                    id=block.id,
                    type="function",
                    function=FunctionCallDelta(name=block.name),
                )
            ],
        )
    return None


def map_content_block_delta(event: "RawContentBlockDeltaEvent") -> ChatChunk | None:
    """Map a content_block_delta event to a ChatChunk."""
    delta = event.delta
    if delta.type == "text_delta":
        return ChatChunk(delta=delta.text)
    if delta.type == "input_json_delta":
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=event.index,
                    function=FunctionCallDelta(arguments=delta.partial_json),
                )
            ],
        )
    if delta.type == "thinking_delta":
        return ChatChunk(reasoning_delta=delta.thinking)
    return None


def map_message_delta(event: "RawMessageDeltaEvent", start_usage: Usage) -> ChatChunk:
    """Map the message_delta event (final event with output usage)."""
    usage = _map_delta_usage(event.usage, start_usage)
    return ChatChunk(
        finish_reason=_map_stop_reason(event.delta.stop_reason),
        usage=usage,
    )


def _map_delta_usage(delta_usage: "MessageDeltaUsage", start_usage: Usage) -> Usage:
    """Combine input tokens from message_start with output tokens from message_delta."""
    return Usage(
        input_tokens=start_usage.input_tokens,
        output_tokens=delta_usage.output_tokens,
        cache_read_tokens=start_usage.cache_read_tokens,
        cache_creation_tokens=start_usage.cache_creation_tokens,
        cache_creation_tokens_by_ttl=start_usage.cache_creation_tokens_by_ttl,
    )
