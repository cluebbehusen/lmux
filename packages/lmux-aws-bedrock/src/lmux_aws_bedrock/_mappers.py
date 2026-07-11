"""Internal mappers between lmux types and Bedrock Converse API types."""

import base64
import copy
import json
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.literals import CacheTTLType, ImageFormatType
    from mypy_boto3_bedrock_runtime.type_defs import (
        CachePointBlockTypeDef,
        ContentBlockTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
    )

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
    EmbeddingResponse,
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
from lmux_aws_bedrock._wire import (
    WireContentBlockDeltaEvent,
    WireContentBlockStartEvent,
    WireConverseResponse,
    WireEmbeddingResponse,
    WireMetadataEvent,
    WireStreamEvent,
    WireTokenUsage,
)

type CostCalculator = Callable[[str, Usage], Cost | None]

PROVIDER_NAME = "aws-bedrock"

_DATA_URI_PATTERN = re.compile(r"^data:image/([^;]+);base64,(.+)$", re.DOTALL)

_STOP_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
}


# MARK: Input Mappers (lmux -> Converse API)


def map_messages(
    messages: Sequence[Message],
) -> tuple[list["SystemContentBlockTypeDef"] | None, list["MessageTypeDef"]]:
    """Convert lmux Messages to Converse API format.

    Returns ``(system_blocks, conversation_messages)`` where ``system_blocks``
    is a list of text content blocks for the ``system`` parameter, and
    ``conversation_messages`` contains only user/assistant role messages.
    """
    system_parts: list[SystemContentBlockTypeDef] = []
    result: list[MessageTypeDef] = []

    for msg in messages:
        if isinstance(msg, SystemMessage | DeveloperMessage):
            system_parts.append({"text": msg.content})
        elif isinstance(msg, UserMessage):
            content, leading_cache_point = _map_user_content(msg.content)
            if leading_cache_point is not None:
                _attach_cache_point(result, system_parts, leading_cache_point)
            if content or not msg.content:
                result.append({"role": "user", "content": content})
        elif isinstance(msg, AssistantMessage):
            result.append(_map_assistant_message(msg))
        else:
            _append_tool_result(result, msg)

    system = system_parts or None
    return system, result


def _map_user_content(
    content: str | list[ContentPart],
) -> tuple[list["ContentBlockTypeDef"], CachePointContent | None]:
    """Map user content, emitting cache points as inline ``cachePoint`` blocks.

    Returns ``(blocks, leading_cache_point)`` — a cache point with no
    preceding block in the same message is returned for the caller to place
    after whatever precedes the message (the prior message, or the system
    blocks), keeping marker-only messages out of the conversation.
    """
    if isinstance(content, str):
        return [{"text": content}], None
    blocks: list[ContentBlockTypeDef] = []
    leading_cache_point: CachePointContent | None = None
    for part in content:
        if isinstance(part, CachePointContent):
            if not blocks:
                if leading_cache_point is None:
                    leading_cache_point = part
            elif "cachePoint" not in blocks[-1]:
                blocks.append({"cachePoint": _map_cache_point(part)})
        else:
            blocks.append(_map_content_part(part))
    return blocks, leading_cache_point


def _attach_cache_point(
    result: list["MessageTypeDef"],
    system_parts: list["SystemContentBlockTypeDef"],
    cache_point: CachePointContent,
) -> None:
    """Place a leading cache point after the previous message, or after the system blocks.

    No-ops when the target has nothing cacheable (empty content) or already
    ends in a cache point (the first marker wins).
    """
    if result:
        blocks = cast("list[ContentBlockTypeDef]", result[-1]["content"])
        if blocks and "cachePoint" not in blocks[-1]:
            blocks.append({"cachePoint": _map_cache_point(cache_point)})
    elif system_parts and "cachePoint" not in system_parts[-1]:
        system_parts.append({"cachePoint": _map_cache_point(cache_point)})
    # else: a cache point at the very start of the request marks an empty prefix


def _map_cache_point(cache_point: CachePointContent) -> "CachePointBlockTypeDef":
    block: CachePointBlockTypeDef = {"type": "default"}
    if cache_point.ttl is not None:
        # ttl is a deliberate passthrough — the service validates the value
        block["ttl"] = cast("CacheTTLType", cache_point.ttl)
    return block


def _map_content_part(part: TextContent | ImageContent) -> "ContentBlockTypeDef":
    if isinstance(part, TextContent):
        return {"text": part.text}
    return _map_image_content(part)


def _map_image_content(img: ImageContent) -> "ContentBlockTypeDef":
    match = _DATA_URI_PATTERN.match(img.url)
    if match:
        fmt = cast("ImageFormatType", match.group(1))
        data = base64.b64decode(match.group(2))
        return {"image": {"format": fmt, "source": {"bytes": data}}}
    msg = "Bedrock Converse API requires base64 data URIs for images, not URLs"
    raise UnsupportedFeatureError(msg, provider=PROVIDER_NAME)


def _map_assistant_message(msg: AssistantMessage) -> "MessageTypeDef":
    content: list[ContentBlockTypeDef] = []
    if msg.content is not None:
        content.append({"text": msg.content})
    if msg.tool_calls:
        content.extend(
            {
                "toolUse": {
                    "toolUseId": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
            }
            for tc in msg.tool_calls
        )
    return {"role": "assistant", "content": content}


def _append_tool_result(result: list["MessageTypeDef"], msg: ToolMessage) -> None:
    """Append a toolResult block, merging consecutive tool results into one user message."""
    tool_block: ContentBlockTypeDef = {
        "toolResult": {
            "toolUseId": msg.tool_call_id,
            "content": [{"text": msg.content}],
            "status": "success",
        }
    }
    if result and result[-1].get("role") == "user":
        last_content = result[-1]["content"]
        if isinstance(last_content, list) and last_content and "toolResult" in last_content[0]:
            last_content.append(tool_block)  # ty: ignore[invalid-argument-type]
            return
    result.append({"role": "user", "content": [tool_block]})


def map_tools(tools: list[Tool]) -> "ToolConfigurationTypeDef":
    """Convert lmux Tools to Converse toolConfig dict."""
    tool_specs: list[ToolTypeDef] = []
    for tool in tools:
        spec: ToolSpecificationTypeDef = {
            "name": tool.function.name,
            "inputSchema": {"json": tool.function.parameters or {"type": "object"}},
        }
        if tool.function.description is not None:
            spec["description"] = tool.function.description
        tool_specs.append({"toolSpec": spec})
    return {"tools": tool_specs}


def map_tool_choice(tc: ToolChoice) -> dict[str, object]:
    """Convert lmux ToolChoice to Bedrock ``toolChoice`` dict."""
    if tc == "none":
        msg = "tool_choice='none' is not supported by Bedrock; omit tools instead"
        raise UnsupportedFeatureError(msg, provider="aws-bedrock")
    if tc == "required":
        return {"any": {}}
    if isinstance(tc, ToolChoiceFunction):
        return {"tool": {"name": tc.name}}
    return {"auto": {}}


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


def map_response_format(rf: ResponseFormat) -> dict[str, object] | None:
    """Convert lmux ResponseFormat to Bedrock ``outputConfig`` fields."""
    if isinstance(rf, TextResponseFormat):
        return None
    if isinstance(rf, JsonObjectResponseFormat):
        msg = "JsonObjectResponseFormat is not supported by Bedrock; use JsonSchemaResponseFormat instead"
        raise UnsupportedFeatureError(msg, provider="aws-bedrock")

    patched = copy.deepcopy(rf.json_schema)
    add_additional_properties_false(patched)
    json_schema: dict[str, str] = {
        "schema": json.dumps(patched, sort_keys=True),
        "name": rf.name,
    }
    if rf.description is not None:
        json_schema["description"] = rf.description

    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": json_schema,
            },
        },
    }


# MARK: Output Mappers (Converse response -> lmux)


def map_converse_response(
    response: WireConverseResponse,
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert a validated Converse API response to an lmux ChatResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_parts: list[str] = []

    for block in response.output.message.content:
        if block.reasoning_content is not None:
            reasoning_text = block.reasoning_content.reasoning_text
            if reasoning_text.text:
                reasoning_parts.append(reasoning_text.text)
        elif block.text is not None:
            text_parts.append(block.text)
        elif block.tool_use is not None:
            tu = block.tool_use
            tool_calls.append(
                ToolCall(
                    id=tu.tool_use_id,
                    function=FunctionCallResult(name=tu.name, arguments=json.dumps(tu.input)),
                )
            )

    content = "\n".join(text_parts) if text_parts else None
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    finish_reason = _map_stop_reason(response.stop_reason)
    usage = _map_token_usage(response.usage) if response.usage else None
    cost = cost_fn(model, usage) if usage else None

    return ChatResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
        finish_reason=finish_reason,
    )


def _map_stop_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return _STOP_REASON_MAP.get(stop_reason, stop_reason)


def _map_token_usage(usage_data: WireTokenUsage) -> Usage:
    cache_read = usage_data.cache_read_input_tokens or 0
    cache_write = usage_data.cache_write_input_tokens or 0
    return Usage(
        # Converse reports inputTokens exclusive of cached tokens; lmux's
        # Usage convention is the inclusive total (see lmux.types.Usage).
        input_tokens=usage_data.input_tokens + cache_read + cache_write,
        output_tokens=usage_data.output_tokens,
        cache_read_tokens=cache_read or None,
        cache_creation_tokens=cache_write or None,
        cache_creation_tokens_by_ttl=_map_cache_details(usage_data),
    )


def _map_cache_details(usage_data: WireTokenUsage) -> dict[str, int] | None:
    """Aggregate the per-TTL cache-write breakdown (``cacheDetails``) if reported."""
    breakdown: dict[str, int] = {}
    for detail in usage_data.cache_details or []:
        if detail.input_tokens:
            breakdown[detail.ttl] = breakdown.get(detail.ttl, 0) + detail.input_tokens
    return breakdown or None


# MARK: Stream Event Mappers


def map_stream_event(event: WireStreamEvent) -> ChatChunk | None:
    """Map a single ConverseStream event to a ChatChunk, or None to skip."""
    if event.content_block_delta is not None:
        return _map_content_block_delta(event.content_block_delta)
    if event.content_block_start is not None:
        return _map_content_block_start(event.content_block_start)
    if event.message_stop is not None:
        return ChatChunk(finish_reason=_map_stop_reason(event.message_stop.stop_reason))
    if event.metadata is not None:
        return _map_metadata_event(event.metadata)
    # messageStart, contentBlockStop — not interesting
    return None


def _map_content_block_delta(data: WireContentBlockDeltaEvent) -> ChatChunk | None:
    delta = data.delta
    if delta.reasoning_content is not None:
        if delta.reasoning_content.text:
            return ChatChunk(reasoning_delta=delta.reasoning_content.text)
        return None
    if delta.text is not None:
        return ChatChunk(delta=delta.text)
    if delta.tool_use is not None:
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=data.content_block_index,
                    function=FunctionCallDelta(arguments=delta.tool_use.input),
                )
            ]
        )
    return None


def _map_content_block_start(data: WireContentBlockStartEvent) -> ChatChunk | None:
    if data.start.tool_use is not None:
        tu = data.start.tool_use
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=data.content_block_index,
                    id=tu.tool_use_id,
                    type="function",
                    function=FunctionCallDelta(name=tu.name),
                )
            ]
        )
    return None


def _map_metadata_event(metadata: WireMetadataEvent) -> ChatChunk | None:
    if metadata.usage is None:
        return None
    return ChatChunk(usage=_map_token_usage(metadata.usage))


# MARK: Embedding Mappers


def build_embedding_request_body(text: str, *, dimensions: int | None = None) -> str:
    """Build the JSON request body for a Titan embedding model."""
    body: dict[str, Any] = {"inputText": text}
    if dimensions is not None:
        body["dimensions"] = dimensions
    return json.dumps(body)


def map_embedding_response(
    response_body: WireEmbeddingResponse,
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert a validated single InvokeModel embedding response to an lmux EmbeddingResponse."""
    usage = Usage(input_tokens=response_body.input_text_token_count, output_tokens=0)
    cost = cost_fn(model, usage)

    return EmbeddingResponse(
        embeddings=[response_body.embedding],
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
    )
