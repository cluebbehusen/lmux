"""Internal mappers between lmux types and Gemini REST JSON.

Input mappers emit plain ``dict`` bodies in the camelCase shape the Gemini REST
API expects (``contents``/``parts``, ``functionDeclarations``, ``toolConfig`` …).
Output mappers consume the raw JSON dicts the API returns (``candidates``,
``usageMetadata`` …) — there is no SDK object in between.
"""

import json
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
from lmux_google._wire import (
    WireBatchEmbeddingsResponse,
    WireCandidate,
    WireEmbedContentResponse,
    WireGenerateContentResponse,
    WireUsageMetadata,
    WireVertexPredictResponse,
)

type CostCalculator = Callable[[str, Usage], Cost | None]
type Json = dict[str, Any]

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


# MARK: Input Mappers (lmux -> Gemini JSON)


def map_messages(messages: Sequence[Message]) -> tuple[str | None, list[Json]]:
    """Convert lmux Messages to Gemini REST format.

    Returns ``(system_instruction, contents)`` where ``system_instruction`` is a
    concatenated string (the caller wraps it in a Content object), and ``contents``
    is the conversation history in Gemini Content dict format.
    """
    system_parts: list[str] = []
    contents: list[Json] = []

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
            if not parts and msg.content:
                continue  # message held only cache points, which this provider has no representation for
            contents.append({"role": "user", "parts": parts})
        elif isinstance(msg, AssistantMessage):
            contents.append(_map_assistant_message(msg))
        else:
            contents.append(_map_tool_message(msg, tool_call_names))

    system = "\n".join(system_parts) if system_parts else None
    return system, contents


def _map_user_content(content: str | list[ContentPart]) -> list[Json]:
    if isinstance(content, str):
        return [{"text": content}]
    # Cache points are dropped: this provider caches implicitly and has no explicit representation.
    return [_map_content_part(part) for part in content if not isinstance(part, CachePointContent)]


def _map_content_part(part: TextContent | ImageContent) -> Json:
    if isinstance(part, TextContent):
        return {"text": part.text}
    return _map_image_content(part)


def _map_image_content(img: ImageContent) -> Json:
    match = _DATA_URI_PATTERN.match(img.url)
    if match:
        mime_type = f"image/{match.group(1)}"
        # REST expects the base64 string as-is (not decoded bytes).
        return {"inlineData": {"data": match.group(2), "mimeType": mime_type}}
    # Plain URL — use fileData (works for GCS URIs and HTTP URLs)
    return {"fileData": {"fileUri": img.url, "mimeType": "image/*"}}


def _map_assistant_message(msg: AssistantMessage) -> Json:
    parts: list[Json] = []
    if msg.content is not None:
        parts.append({"text": msg.content})
    if msg.tool_calls:
        parts.extend(
            {
                "functionCall": {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
            }
            for tc in msg.tool_calls
        )
    return {"role": "model", "parts": parts}


def _map_tool_message(msg: ToolMessage, tool_call_names: dict[str, str]) -> Json:
    name = tool_call_names.get(msg.tool_call_id, msg.tool_call_id)
    try:
        response_data = json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        response_data = {"result": msg.content}
    return {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "id": msg.tool_call_id,
                    "name": name,
                    "response": response_data,
                }
            }
        ],
    }


def map_tools(tools: list[Tool]) -> list[Json]:
    """Convert lmux Tools to Gemini tool dicts."""
    declarations: list[Json] = []
    for tool in tools:
        decl: Json = {"name": tool.function.name}
        if tool.function.description is not None:
            decl["description"] = tool.function.description
        if tool.function.parameters is not None:
            decl["parametersJsonSchema"] = tool.function.parameters
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


def map_tool_choice(tc: ToolChoice) -> Json:
    """Convert lmux ToolChoice to a Gemini ``toolConfig`` dict."""
    if isinstance(tc, ToolChoiceFunction):
        return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [tc.name]}}
    mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}[tc]
    return {"functionCallingConfig": {"mode": mode}}


def map_response_format(rf: ResponseFormat) -> tuple[str | None, Json | None]:
    """Convert lmux ResponseFormat to ``(responseMimeType, responseSchema)`` config fields."""
    if isinstance(rf, TextResponseFormat):
        return None, None
    if isinstance(rf, JsonObjectResponseFormat):
        return "application/json", None
    # JsonSchemaResponseFormat
    return "application/json", rf.json_schema


# MARK: Output Mappers (Gemini JSON -> lmux)


def map_generate_content_response(
    response: WireGenerateContentResponse,
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert a validated Gemini ``generateContent`` response to an lmux ChatResponse."""
    candidate = _get_candidate(response)
    if candidate is None:
        usage = _map_usage(response.usage_metadata)
        cost = cost_fn(model, usage) if usage else None
        return ChatResponse(content=None, tool_calls=None, usage=usage, cost=cost, model=model, provider=provider_name)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    server_tool_results: list[ServerToolResult] = []

    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    pending_code_input: dict[str, str | None] | None = None
    for i, part in enumerate(parts):
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
                    function=FunctionCallResult(name=fc.name or "", arguments=json.dumps(fc.args or {})),
                )
            )
        if part.executable_code is not None:
            ec = part.executable_code
            pending_code_input = {"code": ec.code, "language": ec.language}
        if part.code_execution_result is not None:
            cer = part.code_execution_result
            server_tool_results.append(
                ServerToolResult(
                    name="code_execution",
                    input=pending_code_input,
                    output=cer.output,
                    provider_specific_fields={"outcome": cer.outcome} if cer.outcome else None,
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


def map_generate_content_chunk(
    chunk: WireGenerateContentResponse,
    model: str,
    provider_name: str,
) -> ChatChunk:
    """Convert a validated streaming ``generateContent`` chunk to an lmux ChatChunk."""
    delta: str | None = None
    reasoning_delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    server_tool_deltas: list[ServerToolDelta] | None = None
    finish_reason: str | None = None

    candidate = _get_candidate(chunk)
    if candidate is not None:
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        text_pieces: list[str] = []
        thinking_pieces: list[str] = []
        tcd_list: list[ToolCallDelta] = []
        std_list: list[ServerToolDelta] = []
        std_index = 0
        for i, part in enumerate(parts):
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
                        function=FunctionCallDelta(name=fc.name, arguments=json.dumps(fc.args or {})),
                    )
                )
            if part.executable_code is not None:
                ec = part.executable_code
                std_list.append(
                    ServerToolDelta(
                        index=std_index,
                        name="code_execution",
                        input_delta=json.dumps({"code": ec.code, "language": ec.language}),
                    )
                )
            if part.code_execution_result is not None:
                std_list.append(ServerToolDelta(index=std_index, output_delta=part.code_execution_result.output))
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

    return ChatChunk(
        delta=delta,
        reasoning_delta=reasoning_delta,
        tool_call_deltas=tool_call_deltas,
        server_tool_deltas=server_tool_deltas,
        usage=_map_usage(chunk.usage_metadata),
        finish_reason=finish_reason,
        model=model,
        provider=provider_name,
    )


def map_batch_embeddings_response(
    response: WireBatchEmbeddingsResponse,
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert a validated Gemini ``batchEmbedContents`` response to an lmux EmbeddingResponse."""
    embeddings: list[list[float]] = [list(emb.values or []) for emb in response.embeddings or []]

    # The Developer API reports input tokens in usageMetadata.promptTokenCount; older API versions
    # omit it entirely, in which case token usage (and therefore cost) is left at zero.
    input_tokens = 0
    prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else None
    if prompt_tokens is not None:
        input_tokens = prompt_tokens
    usage = Usage(input_tokens=input_tokens, output_tokens=0)
    cost = cost_fn(model, usage)

    return EmbeddingResponse(
        embeddings=embeddings,
        usage=usage,
        cost=cost,
        model=model,
        provider=provider_name,
    )


def map_embed_content_response(response: WireEmbedContentResponse) -> tuple[list[float], int]:
    """Extract (embedding values, prompt tokens) from one Vertex ``:embedContent`` response."""
    return list(response.embedding.values), response.usage_metadata.prompt_token_count


def map_vertex_embed_response(
    response: WireVertexPredictResponse,
    model: str,
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert a validated Vertex AI ``:predict`` embeddings response to an lmux EmbeddingResponse."""
    embeddings: list[list[float]] = [list(p.embeddings.values) for p in response.predictions]
    input_tokens = sum(p.embeddings.statistics.token_count for p in response.predictions)
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


def _get_candidate(response: WireGenerateContentResponse) -> WireCandidate | None:
    if response.candidates:
        return response.candidates[0]
    return None


def _map_finish_reason(reason: str | None, has_tool_calls: bool) -> str | None:
    if reason is None:
        return None
    if has_tool_calls:
        return "tool_calls"
    return _FINISH_REASON_MAP.get(reason, reason)


def _map_usage(usage_metadata: WireUsageMetadata | None) -> Usage | None:
    if usage_metadata is None:
        return None
    return Usage(
        input_tokens=usage_metadata.prompt_token_count or 0,
        output_tokens=usage_metadata.candidates_token_count or 0,
        cache_read_tokens=usage_metadata.cached_content_token_count or None,
        reasoning_tokens=usage_metadata.thoughts_token_count or None,
    )
