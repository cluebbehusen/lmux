"""Internal mappers between lmux types and Bedrock Converse API types."""

import base64
import json
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.literals import ImageFormatType
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockDeltaEventTypeDef,
        ContentBlockStartEventTypeDef,
        ContentBlockTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
    )

from lmux.exceptions import UnsupportedFeatureError
from lmux.types import (
    AssistantMessage,
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
    ToolMessage,
    Usage,
    UserMessage,
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
            content = _map_user_content(msg.content)
            result.append({"role": "user", "content": content})
        elif isinstance(msg, AssistantMessage):
            result.append(_map_assistant_message(msg))
        else:
            _append_tool_result(result, msg)

    system = system_parts or None
    return system, result


def _map_user_content(content: str | list[ContentPart]) -> list["ContentBlockTypeDef"]:
    if isinstance(content, str):
        return [{"text": content}]
    return [_map_content_part(part) for part in content]


def _map_content_part(part: ContentPart) -> "ContentBlockTypeDef":
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
            last_content.append(tool_block)
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


def _add_additional_properties_false(schema: dict[str, Any]) -> None:
    """Recursively set ``additionalProperties: false`` on all object-typed nodes.

    Bedrock's Converse API requires this field on every ``object`` node in a
    JSON Schema.  Many schema generators omit it, so we patch in-place before
    sending to Bedrock.
    """
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            _add_additional_properties_false(cast("dict[str, Any]", value))
        elif isinstance(value, list):
            for item in cast("list[Any]", value):
                if isinstance(item, dict):
                    _add_additional_properties_false(cast("dict[str, Any]", item))


def map_response_format(rf: ResponseFormat) -> dict[str, object] | None:
    """Convert lmux ResponseFormat to Bedrock ``outputConfig`` fields."""
    if isinstance(rf, TextResponseFormat):
        return None
    if isinstance(rf, JsonObjectResponseFormat):
        msg = "JsonObjectResponseFormat is not supported by Bedrock; use JsonSchemaResponseFormat instead"
        raise UnsupportedFeatureError(msg, provider="aws-bedrock")

    patched = rf.json_schema.copy()
    _add_additional_properties_false(patched)
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
    response: "ConverseResponseTypeDef",
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert Converse API response dict to lmux ChatResponse."""
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_parts: list[str] = []

    for block in content_blocks:
        if "reasoningContent" in block:
            reasoning_text = block["reasoningContent"].get("reasoningText", {})
            if text := reasoning_text.get("text"):
                reasoning_parts.append(text)
        elif "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tu["toolUseId"],
                    function=FunctionCallResult(
                        name=tu["name"],
                        arguments=json.dumps(tu["input"]),
                    ),
                )
            )

    content = "\n".join(text_parts) if text_parts else None
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    stop_reason = response.get("stopReason")
    finish_reason = _map_stop_reason(stop_reason)
    usage = _map_converse_usage(response)
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


def _map_converse_usage(response: "ConverseResponseTypeDef") -> Usage | None:
    usage_data = response.get("usage")
    if usage_data is None:  # pyright: ignore[reportUnnecessaryComparison]
        return None
    cache_read: int | None = usage_data.get("cacheReadInputTokenCount") or None
    cache_write: int | None = usage_data.get("cacheWriteInputTokenCount") or None
    return Usage(
        input_tokens=usage_data.get("inputTokens", 0),
        output_tokens=usage_data.get("outputTokens", 0),
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
    )


# MARK: Stream Event Mappers


def map_stream_event(event: "ConverseStreamOutputTypeDef") -> ChatChunk | None:
    """Map a single ConverseStream event to a ChatChunk, or None to skip."""
    if "contentBlockDelta" in event:
        return _map_content_block_delta(event["contentBlockDelta"])
    if "contentBlockStart" in event:
        return _map_content_block_start(event["contentBlockStart"])
    if "messageStop" in event:
        return ChatChunk(finish_reason=_map_stop_reason(event["messageStop"].get("stopReason")))
    if "metadata" in event:
        return _map_metadata_event(event["metadata"])
    # messageStart, contentBlockStop — not interesting
    return None


def _map_content_block_delta(data: "ContentBlockDeltaEventTypeDef") -> ChatChunk | None:
    delta = data.get("delta", {})
    if "reasoningContent" in delta:
        text = delta["reasoningContent"].get("text")
        if text:
            return ChatChunk(reasoning_delta=text)
        return None
    if "text" in delta:
        return ChatChunk(delta=delta["text"])
    if "toolUse" in delta:
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=data.get("contentBlockIndex", 0),
                    function=FunctionCallDelta(arguments=delta["toolUse"].get("input", "")),
                )
            ]
        )
    return None


def _map_content_block_start(data: "ContentBlockStartEventTypeDef") -> ChatChunk | None:
    start = data.get("start", {})
    if "toolUse" in start:
        tu = start["toolUse"]
        return ChatChunk(
            tool_call_deltas=[
                ToolCallDelta(
                    index=data.get("contentBlockIndex", 0),
                    id=tu.get("toolUseId"),
                    type="function",
                    function=FunctionCallDelta(name=tu.get("name")),
                )
            ]
        )
    return None


def _map_metadata_event(metadata: "ConverseStreamMetadataEventTypeDef") -> ChatChunk | None:
    usage_data = metadata.get("usage")
    if usage_data is None:  # pyright: ignore[reportUnnecessaryComparison]
        return None
    cache_read: int | None = usage_data.get("cacheReadInputTokenCount") or None
    cache_write: int | None = usage_data.get("cacheWriteInputTokenCount") or None
    return ChatChunk(
        usage=Usage(
            input_tokens=usage_data.get("inputTokens", 0),
            output_tokens=usage_data.get("outputTokens", 0),
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_write,
        )
    )


# MARK: Embedding Mappers


def build_embedding_request_body(text: str, *, dimensions: int | None = None) -> str:
    """Build the JSON request body for a Titan embedding model."""
    body: dict[str, Any] = {"inputText": text}
    if dimensions is not None:
        body["dimensions"] = dimensions
    return json.dumps(body)


def map_embedding_response(
    response_body: dict[str, Any],
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert a single InvokeModel embedding response to lmux EmbeddingResponse."""
    embedding: list[float] = response_body.get("embedding", [])
    input_token_count: int = response_body.get("inputTextTokenCount", 0)

    usage = Usage(input_tokens=input_token_count, output_tokens=0)
    cost = cost_fn(model, usage)

    return EmbeddingResponse(
        embeddings=[embedding],
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
    )
