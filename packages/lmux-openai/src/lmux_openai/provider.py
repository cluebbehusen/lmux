"""OpenAI provider implementation (SDK-lite, httpx transport)."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    import httpx

from lmux._http import aiter_sse, iter_sse
from lmux.cost import ModelPricing, calculate_cost
from lmux.exceptions import LmuxError
from lmux.protocols import AuthProvider, CompletionProvider, EmbeddingProvider, PricingProvider, ResponsesProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    ResponseInputItem,
    ResponseResponse,
    Tool,
    ToolChoice,
    Usage,
)
from lmux_openai._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_body,
    raise_for_status,
)
from lmux_openai._lazy import create_async_client, create_sync_client
from lmux_openai._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_response_input,
    map_responses_response,
    map_tool_choice,
    map_tools,
)
from lmux_openai._wire import (
    WireChunk,
    WireCompletion,
    WireEmbeddingResponse,
    WireResponsesResponse,
)
from lmux_openai.auth import OpenAIEnvAuthProvider
from lmux_openai.cost import (
    REGIONAL_UPLIFT,
    apply_cost_multiplier,
    calculate_openai_cost,
    regional_uplift_applies,
)
from lmux_openai.params import OpenAIParams

PROVIDER_NAME = "openai"
_CHAT_PATH = "/chat/completions"
_EMBEDDINGS_PATH = "/embeddings"
_RESPONSES_PATH = "/responses"

_HTTP_ERROR = 400
_SSE_DONE = "[DONE]"


class OpenAIProvider(
    CompletionProvider[OpenAIParams],
    EmbeddingProvider[OpenAIParams],
    ResponsesProvider[OpenAIParams],
    PricingProvider,
):
    """OpenAI API provider over httpx."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        data_residency: bool = False,
        organization: str | None = None,
        project: str | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._auth: AuthProvider[str] = auth or OpenAIEnvAuthProvider()
        self._base_url: str | None = base_url
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._data_residency: bool = data_residency
        self._organization: str | None = organization
        self._project: str | None = project
        self._default_headers: Mapping[str, str] | None = default_headers
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_openai_cost(model, usage)

    def _apply_cost_multipliers(self, cost: Cost | None, model: str) -> Cost | None:
        """Apply the regional-processing uplift when configured for this model."""
        if cost is None:
            return None
        if self._data_residency and regional_uplift_applies(model):
            return apply_cost_multiplier(cost, REGIONAL_UPLIFT)
        return cost

    def _apply_response_multipliers[T: ChatResponse | EmbeddingResponse | ResponseResponse](
        self, response: T, model: str
    ) -> T:
        """Wrap a completed response with any applicable cost multipliers."""
        adjusted = self._apply_cost_multipliers(response.cost, model)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            self._sync_client = create_sync_client(
                api_key=self._auth.get_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
                organization=self._organization,
                project=self._project,
                default_headers=self._default_headers,
            )
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = create_async_client(
                api_key=await self._auth.aget_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
                organization=self._organization,
                project=self._project,
                default_headers=self._default_headers,
            )
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

    # MARK: Chat

    @override
    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
            response = client.post(_CHAT_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_chat_completion(parse_body(response, WireCompletion), PROVIDER_NAME, self._calculate_cost)
        return self._apply_response_multipliers(mapped, mapped.model)

    @override
    async def achat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
            response = await client.post(_CHAT_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_chat_completion(parse_body(response, WireCompletion), PROVIDER_NAME, self._calculate_cost)
        return self._apply_response_multipliers(mapped, mapped.model)

    @override
    def chat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> Iterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            with client.stream("POST", _CHAT_PATH, json=body) as response:
                if response.status_code >= _HTTP_ERROR:
                    response.read()
                    raise error_from_response(response)  # noqa: TRY301
                for _event, data in iter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

    @override
    async def achat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            async with client.stream("POST", _CHAT_PATH, json=body) as response:
                if response.status_code >= _HTTP_ERROR:
                    await response.aread()
                    raise error_from_response(response)  # noqa: TRY301
                async for _event, data in aiter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

    # MARK: Embeddings

    @override
    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> EmbeddingResponse:
        body = self._embed_body(model, input, dimensions, provider_params)
        try:
            client = self._get_sync_client()
            response = client.post(_EMBEDDINGS_PATH, json=body)
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_embedding_response(
            parse_body(response, WireEmbeddingResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(mapped, mapped.model)

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> EmbeddingResponse:
        body = self._embed_body(model, input, dimensions, provider_params)
        try:
            client = await self._get_async_client()
            response = await client.post(_EMBEDDINGS_PATH, json=body)
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_embedding_response(
            parse_body(response, WireEmbeddingResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(mapped, mapped.model)

    # MARK: Responses API

    @override
    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        body = self._responses_body(model, input, provider_params)
        try:
            client = self._get_sync_client()
            response = client.post(_RESPONSES_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_responses_response(
            parse_body(response, WireResponsesResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(mapped, mapped.model)

    @override
    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        body = self._responses_body(model, input, provider_params)
        try:
            client = await self._get_async_client()
            response = await client.post(_RESPONSES_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        mapped = map_responses_response(
            parse_body(response, WireResponsesResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(mapped, mapped.model)

    # MARK: Internal Helpers

    def _map_stream_chunk(self, chunk: dict[str, Any], model: str) -> ChatChunk:
        wire = WireChunk.model_validate(chunk)
        mapped = map_chat_chunk(wire, PROVIDER_NAME)
        if mapped.usage is not None:
            cost_model = wire.model or model
            cost = self._apply_cost_multipliers(self._calculate_cost(cost_model, mapped.usage), cost_model)
            mapped = mapped.model_copy(update={"cost": cost})
        return mapped

    @staticmethod
    def _embed_body(
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: OpenAIParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            body["dimensions"] = dimensions
        if provider_params is not None:
            body.update(OpenAIProvider._provider_params_kwargs(provider_params))
        return body

    @staticmethod
    def _responses_body(
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        provider_params: OpenAIParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": map_response_input(input)}
        if provider_params is not None:
            body.update(OpenAIProvider._provider_params_kwargs(provider_params))
            # Responses API uses reasoning={"effort": ...}, not flat reasoning_effort
            if provider_params.reasoning_effort is not None:
                body["reasoning"] = {"effort": provider_params.reasoning_effort}
        return body

    def _stream_body(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        provider_params: OpenAIParams | None,
    ) -> dict[str, Any]:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        return {**body, "stream": True, "stream_options": {"include_usage": True}}

    @staticmethod
    def _build_body(  # noqa: PLR0913
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        provider_params: OpenAIParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": map_messages(messages)}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            if model.startswith(("gpt-5", "o1", "o3", "o4")):
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if tools is not None:
            body["tools"] = map_tools(tools)
        if tool_choice is not None:
            body["tool_choice"] = map_tool_choice(tool_choice)
        if response_format is not None:
            body["response_format"] = map_response_format(response_format)
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if provider_params is not None:
            body.update(OpenAIProvider._provider_params_kwargs(provider_params))
            # Chat Completions uses flat reasoning_effort; provider_params overrides top-level
            if provider_params.reasoning_effort is not None:
                body["reasoning_effort"] = provider_params.reasoning_effort
        return body

    @staticmethod
    def _provider_params_kwargs(params: OpenAIParams) -> dict[str, Any]:
        """Convert OpenAIParams to kwargs shared across all OpenAI API surfaces.

        Reasoning is intentionally excluded here because the Chat Completions API
        and Responses API use different field shapes (``reasoning_effort`` vs
        ``reasoning``).  Each call site maps it separately.
        """
        kwargs: dict[str, Any] = {}
        if params.service_tier is not None:
            kwargs["service_tier"] = params.service_tier
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs
