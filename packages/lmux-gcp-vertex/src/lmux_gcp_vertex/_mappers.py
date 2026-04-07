"""Internal mappers between lmux types and google-genai types."""

import base64
import json
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai.types import (
        Candidate,
        ContentDict,
        EmbedContentResponse,
        FinishReason,
        FunctionDeclarationDict,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        PartDict,
        ToolDict,
    )

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
    ServerToolDelta,
    ServerToolResult,
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

type CostCalculator = Callable[[str, Usage], Cost | None]

_DATA_URI_PATTERN = re.compile(r"^data:image/([^;]+);base64,(.+)$", re.DOTALL)

_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "IMAGE_RECITATION": "content_filter",
    "LANGUAGE": "content_filter",
}


# MARK: Input Mappers (lmux -> google-genai)


def map_messages(messages: Sequence[Message]) -> tuple[str | None, list["ContentDict"]]:
    """Convert lmux Messages to google-genai format.

    Returns ``(system_instruction, contents)`` where ``system_instruction``
    is a concatenated string for the config parameter, and ``contents`` is
    the conversation history in google-genai Content dict format.
    """
    system_parts: list[str] = []
    contents: list[ContentDict] = []

    # Build tool_call_id -> function_name mapping for ToolMessage translation
    tool_call_names: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_names[tc.id] = tc.function.name

    for msg in messages:
        if isinstance(msg, SystemMessage | DeveloperMessage):
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            parts = _map_user_content(msg.content)
            contents.append({"role": "user", "parts": parts})
        elif isinstance(msg, AssistantMessage):
            contents.append(_map_assistant_message(msg))
        else:
            contents.append(_map_tool_message(msg, tool_call_names))

    system = "\n".join(system_parts) if system_parts else None
    return system, contents


def _map_user_content(content: str | list[ContentPart]) -> list["PartDict"]:
    if isinstance(content, str):
        return [{"text": content}]
    return [_map_content_part(part) for part in content]


def _map_content_part(part: ContentPart) -> "PartDict":
    if isinstance(part, TextContent):
        return {"text": part.text}
    return _map_image_content(part)


def _map_image_content(img: ImageContent) -> "PartDict":
    match = _DATA_URI_PATTERN.match(img.url)
    if match:
        mime_type = f"image/{match.group(1)}"
        data = base64.b64decode(match.group(2))
        return {"inline_data": {"data": data, "mime_type": mime_type}}
    # Plain URL — use file_data (works for GCS URIs and HTTP URLs)
    return {"file_data": {"file_uri": img.url, "mime_type": "image/*"}}


def _map_assistant_message(msg: AssistantMessage) -> "ContentDict":
    parts: list[PartDict] = []
    if msg.content is not None:
        parts.append({"text": msg.content})
    if msg.tool_calls:
        parts.extend(
            {
                "function_call": {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
            }
            for tc in msg.tool_calls
        )
    return {"role": "model", "parts": parts}


def _map_tool_message(msg: ToolMessage, tool_call_names: dict[str, str]) -> "ContentDict":
    name = tool_call_names.get(msg.tool_call_id, msg.tool_call_id)
    try:
        response_data = json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        response_data = {"result": msg.content}
    return {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": msg.tool_call_id,
                    "name": name,
                    "response": response_data,
                }
            }
        ],
    }


def map_tools(tools: list[Tool]) -> list["ToolDict"]:
    """Convert lmux Tools to google-genai tool dicts."""
    declarations: list[FunctionDeclarationDict] = []
    for tool in tools:
        decl: FunctionDeclarationDict = {"name": tool.function.name}
        if tool.function.description is not None:
            decl["description"] = tool.function.description
        if tool.function.parameters is not None:
            decl["parameters_json_schema"] = tool.function.parameters
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def map_tool_choice(tc: ToolChoice) -> dict[str, object]:
    """Convert lmux ToolChoice to google-genai ``tool_config`` dict."""
    if isinstance(tc, ToolChoiceFunction):
        return {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tc.name]}}
    mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}[tc]
    return {"function_calling_config": {"mode": mode}}


def map_response_format(rf: ResponseFormat) -> tuple[str | None, dict[str, object] | None]:
    """Convert lmux ResponseFormat to google-genai config fields.

    Returns ``(response_mime_type, response_schema)`` to be merged into
    the ``GenerateContentConfig``.
    """
    if isinstance(rf, TextResponseFormat):
        return None, None
    if isinstance(rf, JsonObjectResponseFormat):
        return "application/json", None
    # JsonSchemaResponseFormat
    return "application/json", rf.json_schema


# MARK: Output Mappers (google-genai -> lmux)


def map_generate_content_response(
    response: "GenerateContentResponse",
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert google-genai GenerateContentResponse to lmux ChatResponse."""
    candidate = _get_candidate(response)
    if candidate is None:
        usage = _map_usage(response.usage_metadata)
        cost = cost_fn(model, usage) if usage else None
        return ChatResponse(content=None, tool_calls=None, usage=usage, cost=cost, model=model, provider=provider_name)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    server_tool_results: list[ServerToolResult] = []

    if candidate.content and candidate.content.parts:
        pending_code_input: dict[str, str | None] | None = None
        for i, part in enumerate(candidate.content.parts):
            if part.thought:
                if part.text is not None:
                    thinking_parts.append(part.text)
                continue
            if part.text is not None:
                text_parts.append(part.text)
            if part.function_call is not None:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=fc.id or f"call_{i}",
                        function=FunctionCallResult(
                            name=fc.name or "",
                            arguments=json.dumps(dict(fc.args) if fc.args else {}),
                        ),
                    )
                )
            ec = getattr(part, "executable_code", None)
            if ec is not None:
                pending_code_input = {
                    "code": ec.code,
                    "language": ec.language.value if hasattr(ec.language, "value") else ec.language,
                }
            cer = getattr(part, "code_execution_result", None)
            if cer is not None:
                outcome = cer.outcome.value if hasattr(cer.outcome, "value") else cer.outcome
                server_tool_results.append(
                    ServerToolResult(
                        name="code_execution",
                        input=pending_code_input,
                        output=cer.output,
                        provider_specific_fields={"outcome": outcome} if outcome else None,
                    )
                )
                pending_code_input = None

    content = "\n".join(text_parts) if text_parts else None
    reasoning = "\n".join(thinking_parts) if thinking_parts else None
    finish_reason = _map_finish_reason(candidate.finish_reason, bool(tool_calls))
    usage = _map_usage(response.usage_metadata)
    cost = cost_fn(model, usage) if usage else None

    return ChatResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or None,
        server_tool_results=server_tool_results or None,
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
        finish_reason=finish_reason,
    )


def map_generate_content_chunk(  # noqa: PLR0912
    chunk: "GenerateContentResponse",
    model: str,
) -> ChatChunk:
    """Convert a streaming chunk to lmux ChatChunk."""
    delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    server_tool_deltas: list[ServerToolDelta] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None

    reasoning_delta: str | None = None
    candidate = _get_candidate(chunk)
    if candidate is not None:
        if candidate.content and candidate.content.parts:
            text_pieces: list[str] = []
            thinking_pieces: list[str] = []
            tcd_list: list[ToolCallDelta] = []
            std_list: list[ServerToolDelta] = []
            std_index = 0
            for i, part in enumerate(candidate.content.parts):
                if part.thought:
                    if part.text is not None:
                        thinking_pieces.append(part.text)
                    continue
                if part.text is not None:
                    text_pieces.append(part.text)
                if part.function_call is not None:
                    fc = part.function_call
                    tcd_list.append(
                        ToolCallDelta(
                            index=i,
                            id=fc.id or f"call_{i}",
                            type="function",
                            function=FunctionCallDelta(
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args) if fc.args else {}),
                            ),
                        )
                    )
                ec = getattr(part, "executable_code", None)
                if ec is not None:
                    language = ec.language.value if hasattr(ec.language, "value") else ec.language
                    std_list.append(
                        ServerToolDelta(
                            index=std_index,
                            name="code_execution",
                            input_delta=json.dumps({"code": ec.code, "language": language}),
                        )
                    )
                cer = getattr(part, "code_execution_result", None)
                if cer is not None:
                    std_list.append(
                        ServerToolDelta(
                            index=std_index,
                            output_delta=cer.output,
                        )
                    )
                    std_index += 1
            if text_pieces:
                delta = "".join(text_pieces)
            if thinking_pieces:
                reasoning_delta = "".join(thinking_pieces)
            if tcd_list:
                tool_call_deltas = tcd_list
            if std_list:
                server_tool_deltas = std_list
        finish_reason = _map_finish_reason(candidate.finish_reason, tool_call_deltas is not None)

    usage = _map_usage(chunk.usage_metadata)

    return ChatChunk(
        delta=delta,
        reasoning_delta=reasoning_delta,
        tool_call_deltas=tool_call_deltas,
        server_tool_deltas=server_tool_deltas,
        usage=usage,
        finish_reason=finish_reason,
        model=model,
    )


def map_embed_content_response(
    response: "EmbedContentResponse",
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert google-genai EmbedContentResponse to lmux EmbeddingResponse."""
    embeddings: list[list[float]] = (
        [list(emb.values) if emb.values else [] for emb in response.embeddings] if response.embeddings else []
    )

    # The embedding API does not return token counts — only billable_character_count
    # in metadata (Vertex AI only). We approximate tokens as chars / 4, consistent
    # with how litellm handles this. This is an approximation, not exact token usage.
    input_tokens = 0
    if response.metadata is not None and response.metadata.billable_character_count is not None:
        input_tokens = response.metadata.billable_character_count // 4
    usage = Usage(input_tokens=input_tokens, output_tokens=0)
    cost = cost_fn(model, usage)

    return EmbeddingResponse(
        embeddings=embeddings,
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
    )


# MARK: Internal Helpers


def _get_candidate(response: "GenerateContentResponse") -> "Candidate | None":
    if response.candidates:
        return response.candidates[0]
    return None


def _map_finish_reason(reason: "FinishReason | None", has_tool_calls: bool) -> str | None:
    if reason is None:
        return None
    if has_tool_calls:
        return "tool_calls"
    reason_str: str = reason.value if hasattr(reason, "value") else str(reason)
    return _FINISH_REASON_MAP.get(reason_str, reason_str)


def _map_usage(usage_metadata: "GenerateContentResponseUsageMetadata | None") -> Usage | None:
    if usage_metadata is None:
        return None
    input_tokens = usage_metadata.prompt_token_count or 0
    output_tokens = usage_metadata.candidates_token_count or 0
    cache_read = usage_metadata.cached_content_token_count or None
    reasoning_tokens = getattr(usage_metadata, "thoughts_token_count", None) or None
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        reasoning_tokens=reasoning_tokens,
    )
