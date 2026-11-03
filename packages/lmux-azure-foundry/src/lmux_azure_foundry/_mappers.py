"""Internal mappers between lmux types and Azure AI Foundry (OpenAI-compatible) JSON.

Input mappers emit plain JSON dicts for the request body; output mappers consume
the raw JSON dicts returned by the REST API (not SDK objects).
"""

import copy
from collections.abc import Callable, Sequence
from typing import Any, cast

from lmux.exceptions import InvalidRequestError
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
    ProviderContinuation,
    ResponseFormat,
    ResponseInputFunctionCall,
    ResponseInputFunctionCallOutput,
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
    ToolMessage,
    Usage,
    UserMessage,
)
from lmux_azure_foundry._wire import (
    WireChunk,
    WireCompletion,
    WireCompletionUsage,
    WireEmbeddingResponse,
    WireOutputItem,
    WireResponsesResponse,
    WireResponsesStreamEvent,
    WireResponsesUsage,
    WireToolCallDelta,
)

type CostCalculator = Callable[[str, Usage], Cost | None]
type Json = dict[str, Any]

_CONTINUATION_NAMESPACE = "lmux_azure_foundry.responses"
_INCOMPLETE_FINISH_REASONS: dict[str, str] = {"max_output_tokens": "length", "content_filter": "content_filter"}


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
    return [mapped for item in input if (mapped := _map_response_input_item(item)) is not None]


def _map_response_input_item(item: ResponseInputItem) -> Json | None:
    """Map one input item; None for a message that held only cache points."""
    if not isinstance(item, ResponseInputMessage):
        return item.model_dump(exclude_none=True)
    if isinstance(item.content, str):
        return {"role": item.role, "content": item.content}
    content = [_map_response_content_part(part) for part in item.content if not isinstance(part, CachePointContent)]
    if content or not item.content:
        return {"role": item.role, "content": content}
    return None


def _map_response_content_part(part: TextContent | ImageContent) -> Json:
    if isinstance(part, TextContent):
        return {"type": "input_text", "text": part.text}
    return {"type": "input_image", "image_url": part.url, "detail": part.detail}


def map_messages_to_response_input(messages: Sequence[Message]) -> list[Json]:
    """Convert lmux chat Messages to Responses API input items.

    An assistant turn that carries this provider's continuation is replayed as the output items the
    API returned for it (reasoning, function calls, message), so the model keeps its reasoning across
    a tool loop. Other assistant turns are rebuilt from their text and tool calls.
    """
    result: list[Json] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage) and (replayed := _replayed_output(msg)) is not None:
            result.extend(replayed)
            continue
        result.extend(
            mapped for item in _to_response_input_items(msg) if (mapped := _map_response_input_item(item)) is not None
        )
    return result


def _replayed_output(msg: AssistantMessage) -> list[Json] | None:
    continuation = msg.continuation
    if continuation is None or continuation.namespace != _CONTINUATION_NAMESPACE:
        return None
    items = continuation.data.get("output")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        error_message = "Azure Foundry continuation data must contain a list of output items"
        raise InvalidRequestError(error_message, provider="azure-foundry")
    return copy.deepcopy(cast("list[Json]", items))


def _to_response_input_items(msg: Message) -> list[ResponseInputItem]:
    if isinstance(msg, SystemMessage | DeveloperMessage | UserMessage):
        return [ResponseInputMessage(role=msg.role, content=msg.content)]
    if isinstance(msg, ToolMessage):
        return [ResponseInputFunctionCallOutput(call_id=msg.tool_call_id, output=msg.content)]
    items: list[ResponseInputItem] = []
    if msg.content:
        items.append(ResponseInputMessage(role="assistant", content=msg.content))
    items.extend(
        ResponseInputFunctionCall(call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in msg.tool_calls or []
    )
    return items


def map_response_tools(tools: list[Tool]) -> list[Json]:
    """Convert lmux Tools to Responses API function tools.

    The Responses API treats an unset ``strict`` as true, unlike Chat Completions, so it is sent as
    false unless the caller set it.
    """
    result: list[Json] = []
    for tool in tools:
        fn: Json = {"type": "function", "name": tool.function.name, "strict": bool(tool.function.strict)}
        if tool.function.description is not None:
            fn["description"] = tool.function.description
        if tool.function.parameters is not None:
            fn["parameters"] = tool.function.parameters
        result.append(fn)
    return result


def map_response_tool_choice(tc: ToolChoice) -> str | Json:
    """Convert lmux ToolChoice to the Responses API tool_choice param."""
    if isinstance(tc, ToolChoiceFunction):
        return {"type": "function", "name": tc.name}
    return tc


def map_response_text_format(rf: ResponseFormat) -> Json:
    """Convert lmux ResponseFormat to the Responses API ``text.format`` param."""
    if isinstance(rf, TextResponseFormat | JsonObjectResponseFormat):
        return {"type": rf.type}
    patched = copy.deepcopy(rf.json_schema)
    add_additional_properties_false(patched)
    fmt: Json = {"type": "json_schema", "name": rf.name, "schema": patched}
    if rf.description is not None:
        fmt["description"] = rf.description
    if rf.strict is not None:
        fmt["strict"] = rf.strict
    return fmt


# MARK: Output Mappers (Azure Foundry wire models -> lmux)


def map_chat_completion(completion: WireCompletion, provider_name: str, cost_fn: CostCalculator) -> ChatResponse:
    """Convert a validated OpenAI-compatible chat completion to an lmux ChatResponse."""
    choice = completion.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        # Keep only function-typed tool calls; other types (e.g. custom) are dropped.
        tool_calls = [
            ToolCall(id=tc.id, function=FunctionCallResult(name=tc.function.name, arguments=tc.function.arguments))
            for tc in message.tool_calls
            if tc.type == "function" and tc.function is not None
        ]

    usage = _map_usage(completion.usage)
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


def _map_usage(usage: WireCompletionUsage | None) -> Usage | None:
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
    """Convert a validated OpenAI-compatible streaming chunk to an lmux ChatChunk."""
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


def map_embedding_response(
    response: WireEmbeddingResponse, provider_name: str, cost_fn: CostCalculator
) -> EmbeddingResponse:
    """Convert a validated OpenAI-compatible embeddings response to an lmux EmbeddingResponse."""
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
    """Convert a validated OpenAI-compatible Responses API response to an lmux ResponseResponse."""
    usage = _map_responses_usage(response.usage)
    cost = cost_fn(response.model, usage) if usage else None
    return ResponseResponse(
        id=response.id,
        output_text=_extract_output_text(response.output or []),
        usage=usage,
        cost=cost,
        model=response.model,
        provider=provider_name,
    )


def _map_responses_usage(usage: WireResponsesUsage | None) -> Usage | None:
    if usage is None:
        return None
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=(input_details.cached_tokens if input_details else None) or None,
        reasoning_tokens=(output_details.reasoning_tokens if output_details else None) or None,
    )


def _extract_output_text(output: list[WireOutputItem]) -> str:
    """Aggregate all ``output_text`` blocks from the Responses API ``output`` list."""
    return "".join(
        content.text or ""
        for item in output
        if item.type == "message"
        for content in item.content or []
        if content.type == "output_text"
    )


def map_responses_chat(response: WireResponsesResponse, provider_name: str, cost_fn: CostCalculator) -> ChatResponse:
    """Convert a Responses API response to an lmux ChatResponse.

    When the output holds reasoning items, the whole output is kept as the continuation, since the
    API needs reasoning items replayed alongside the function calls that followed them.
    """
    output = response.output or []
    tool_calls = [
        ToolCall(id=item.call_id, function=FunctionCallResult(name=item.name, arguments=item.arguments or ""))
        for item in output
        if item.type == "function_call" and item.call_id is not None and item.name is not None
    ]
    summary = "\n\n".join(
        part.text for item in output if item.type == "reasoning" for part in item.summary or [] if part.text
    )
    continuation = None
    if any(item.type == "reasoning" for item in output):
        continuation = ProviderContinuation(
            namespace=_CONTINUATION_NAMESPACE,
            data={"output": [item.model_dump(mode="json", exclude_none=True) for item in output]},
        )
    usage = _map_responses_usage(response.usage)
    return ChatResponse(
        content=_extract_output_text(output) or None,
        reasoning=summary or None,
        tool_calls=tool_calls or None,
        usage=usage,
        cost=cost_fn(response.model, usage) if usage else None,
        model=response.model,
        provider=provider_name,
        finish_reason=_responses_finish_reason(response, has_tool_calls=bool(tool_calls)),
        continuation=continuation,
    )


def _responses_finish_reason(response: WireResponsesResponse, *, has_tool_calls: bool) -> str:
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        return _INCOMPLETE_FINISH_REASONS.get(reason or "", reason or "incomplete")
    return "tool_calls" if has_tool_calls else "stop"


class ResponsesStreamMapper:
    """Map Responses API stream events to lmux ChatChunks, one stream per instance.

    Function calls are numbered in the order they start, which is the tool-call index
    the chunk accumulators expect. The terminal event carries usage, the finish reason
    and the continuation; it is the only chunk with ``cost`` set.
    """

    def __init__(self, provider_name: str, cost_fn: CostCalculator) -> None:
        self._provider_name = provider_name
        self._cost_fn = cost_fn
        self._tool_index: dict[int, int] = {}
        self._summary_parts = 0

    def map_event(self, event: WireResponsesStreamEvent) -> ChatChunk | None:
        """Return the chunk for one stream event, or None for events that carry nothing to emit."""
        kind = event.type
        if kind == "response.output_text.delta":
            return ChatChunk(delta=event.delta, provider=self._provider_name)
        if kind == "response.reasoning_summary_part.added":
            # Summary parts are joined with a blank line, as in the non-streamed response.
            self._summary_parts += 1
            return ChatChunk(reasoning_delta="\n\n", provider=self._provider_name) if self._summary_parts > 1 else None
        if kind == "response.reasoning_summary_text.delta":
            return ChatChunk(reasoning_delta=event.delta, provider=self._provider_name)
        if kind in {"response.completed", "response.incomplete"} and event.response is not None:
            return self._final_chunk(event.response)
        delta = self._tool_call_delta(event)
        return ChatChunk(tool_call_deltas=[delta], provider=self._provider_name) if delta is not None else None

    def _tool_call_delta(self, event: WireResponsesStreamEvent) -> ToolCallDelta | None:
        item = event.item
        if event.type == "response.output_item.added" and item is not None and item.type == "function_call":
            return ToolCallDelta(
                index=self._tool_index.setdefault(event.output_index or 0, len(self._tool_index)),
                id=item.call_id,
                type="function",
                function=FunctionCallDelta(name=item.name, arguments=item.arguments or None),
            )
        if event.type == "response.function_call_arguments.delta" and event.output_index in self._tool_index:
            return ToolCallDelta(
                index=self._tool_index[event.output_index], function=FunctionCallDelta(arguments=event.delta)
            )
        return None

    def _final_chunk(self, response: WireResponsesResponse) -> ChatChunk:
        done = map_responses_chat(response, self._provider_name, self._cost_fn)
        return ChatChunk(
            usage=done.usage,
            cost=done.cost,
            finish_reason=done.finish_reason,
            model=done.model,
            provider=self._provider_name,
            continuation=done.continuation,
        )
