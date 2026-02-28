"""Internal mappers between lmux types and OpenAI SDK types."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

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
    Usage,
    UserMessage,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.responses import Response as OAIResponse

type CostCalculator = Callable[[str, Usage], Cost | None]


# MARK: Input Mappers (lmux -> OpenAI SDK params)


def map_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert lmux Messages to OpenAI-compatible message dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, DeveloperMessage):
            result.append({"role": "developer", "content": msg.content})
        elif isinstance(msg, UserMessage):
            content = _map_user_content(msg.content)
            result.append({"role": "user", "content": content})
        elif isinstance(msg, AssistantMessage):
            d: dict[str, Any] = {"role": "assistant"}
            if msg.content is not None:
                d["content"] = msg.content
            if msg.tool_calls:
                d["tool_calls"] = [_map_tool_call_param(tc) for tc in msg.tool_calls]
            result.append(d)
        else:
            result.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
    return result


def _map_tool_call_param(tc: ToolCall) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _map_user_content(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_map_content_part(part) for part in content]


def _map_content_part(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.text}
    return {"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}


def map_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert lmux Tools to OpenAI tool param dicts."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        fn: dict[str, Any] = {"name": tool.function.name}
        if tool.function.description is not None:
            fn["description"] = tool.function.description
        if tool.function.parameters is not None:
            fn["parameters"] = tool.function.parameters
        if tool.function.strict is not None:
            fn["strict"] = tool.function.strict
        result.append({"type": "function", "function": fn})
    return result


def map_response_format(rf: ResponseFormat) -> dict[str, Any]:
    """Convert lmux ResponseFormat to OpenAI response_format param dict."""
    if isinstance(rf, TextResponseFormat):
        return {"type": "text"}
    if isinstance(rf, JsonObjectResponseFormat):
        return {"type": "json_object"}
    schema_dict: dict[str, Any] = {"name": rf.name, "schema": rf.json_schema}
    if rf.description is not None:
        schema_dict["description"] = rf.description
    if rf.strict is not None:
        schema_dict["strict"] = rf.strict
    return {"type": "json_schema", "json_schema": schema_dict}


def map_response_input(input: str | Sequence[ResponseInputItem]) -> Any:  # noqa: A002, ANN401
    """Convert lmux ResponseInputItem sequence to OpenAI-compatible dicts.

    Returns ``Any`` because the OpenAI SDK expects its own TypedDict union
    (``ResponseInputItemParam``), which is structurally compatible with the
    dicts produced by ``model_dump()`` but not assignable due to nominal typing.
    """
    if isinstance(input, str):
        return input
    return [item.model_dump(exclude_none=True) for item in input]


# MARK: Output Mappers (OpenAI SDK responses -> lmux)


def _map_function_tool_call(tc: "ChatCompletionMessageFunctionToolCall") -> ToolCall:
    return ToolCall(
        id=tc.id,
        function=FunctionCallResult(name=tc.function.name, arguments=tc.function.arguments),
    )


def map_chat_completion(
    completion: "ChatCompletion",
    provider_name: str,
    cost_fn: CostCalculator,
) -> ChatResponse:
    """Convert OpenAI ChatCompletion to lmux ChatResponse."""
    choice = completion.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        tool_calls = [_map_function_tool_call(tc) for tc in message.tool_calls if tc.type == "function"]

    usage = _map_completion_usage(completion)
    cost = cost_fn(completion.model, usage) if usage else None

    return ChatResponse(
        content=message.content,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost,
        model=completion.model,
        provider=provider_name,
        finish_reason=choice.finish_reason,
    )


def _map_completion_usage(completion: "ChatCompletion") -> Usage | None:
    """Extract Usage from a ChatCompletion, or None if the SDK omits it."""
    oai_usage = completion.usage
    if oai_usage is None:
        return None

    cache_read = None
    if oai_usage.prompt_tokens_details and oai_usage.prompt_tokens_details.cached_tokens:
        cache_read = oai_usage.prompt_tokens_details.cached_tokens

    return Usage(
        input_tokens=oai_usage.prompt_tokens,
        output_tokens=oai_usage.completion_tokens,
        cache_read_tokens=cache_read,
    )


def map_chat_chunk(chunk: "ChatCompletionChunk") -> ChatChunk:
    """Convert an OpenAI ChatCompletionChunk to an lmux ChatChunk."""
    delta_text: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    finish_reason: str | None = None

    if chunk.choices:
        choice = chunk.choices[0]
        delta_text = choice.delta.content
        finish_reason = choice.finish_reason

        if choice.delta.tool_calls:
            tool_call_deltas = [
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    type="function" if tc.type == "function" else None,
                    function=FunctionCallDelta(
                        name=tc.function.name if tc.function else None,
                        arguments=tc.function.arguments if tc.function else None,
                    )
                    if tc.function
                    else None,
                )
                for tc in choice.delta.tool_calls
            ]

    usage: Usage | None = None
    if chunk.usage:
        cache_read = None
        if chunk.usage.prompt_tokens_details and chunk.usage.prompt_tokens_details.cached_tokens:
            cache_read = chunk.usage.prompt_tokens_details.cached_tokens
        usage = Usage(
            input_tokens=chunk.usage.prompt_tokens,
            output_tokens=chunk.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

    return ChatChunk(
        delta=delta_text,
        tool_call_deltas=tool_call_deltas,
        usage=usage,
        finish_reason=finish_reason,
        model=chunk.model,
    )


def map_embedding_response(
    response: "CreateEmbeddingResponse",
    provider_name: str,
    cost_fn: CostCalculator,
) -> EmbeddingResponse:
    """Convert OpenAI CreateEmbeddingResponse to lmux EmbeddingResponse."""
    embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    usage = Usage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=0,
    )
    cost = cost_fn(response.model, usage)
    return EmbeddingResponse(
        embeddings=embeddings,
        usage=usage,
        cost=cost,
        model=response.model,
        provider=provider_name,
    )


def map_responses_response(
    response: "OAIResponse",
    provider_name: str,
    cost_fn: CostCalculator,
) -> ResponseResponse:
    """Convert OpenAI Responses API Response to lmux ResponseResponse."""
    usage_data = response.usage
    usage: Usage | None = None
    if usage_data:
        cache_read: int | None = None
        if usage_data.input_tokens_details and usage_data.input_tokens_details.cached_tokens:
            cache_read = usage_data.input_tokens_details.cached_tokens
        usage = Usage(
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            cache_read_tokens=cache_read,
        )

    cost = cost_fn(response.model, usage) if usage else None
    return ResponseResponse(
        id=response.id,
        output_text=response.output_text,
        usage=usage,
        cost=cost,
        model=response.model,
        provider=provider_name,
    )
