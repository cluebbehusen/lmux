"""Azure AI Foundry provider implementation (SDK-lite, httpx transport)."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
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
from lmux_azure_foundry._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_body,
    raise_for_status,
)
from lmux_azure_foundry._lazy import auth_headers, create_async_client, create_sync_client
from lmux_azure_foundry._mappers import (
    has_cache_breakpoint,
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_response_input,
    map_responses_response,
    map_tool_choice,
    map_tools,
    supports_explicit_prompt_cache,
)
from lmux_azure_foundry._wire import (
    WireChunk,
    WireCompletion,
    WireEmbeddingResponse,
    WireResponsesResponse,
)
from lmux_azure_foundry.auth import AzureFoundryCredential, AzureFoundryKeyAuthProvider
from lmux_azure_foundry.cost import (
    DATA_ZONE_MULTIPLIER,
    REGIONAL_MULTIPLIER,
    apply_cost_multiplier,
    calculate_azure_foundry_cost,
)
from lmux_azure_foundry.params import AzureFoundryParams

PROVIDER_NAME = "azure-foundry"
# Minimum version that supports the Responses API (versions are cumulative).
DEFAULT_API_VERSION = "2025-04-01-preview"

_RESPONSES_PATH = "/responses"
_HTTP_ERROR = 400
_SSE_DONE = "[DONE]"
# Models that use max_completion_tokens instead of max_tokens.
_MAX_COMPLETION_TOKEN_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class AzureFoundryProvider(
    CompletionProvider[AzureFoundryParams],
    EmbeddingProvider[AzureFoundryParams],
    ResponsesProvider[AzureFoundryParams],
    PricingProvider,
):
    """Azure AI Foundry API provider over httpx (OpenAI-compatible endpoints).

    Talks to models deployed in Azure AI Foundry / Azure OpenAI directly over
    the REST API. Authentication supports all three Azure methods:

    - **API key** — pass ``auth=AzureFoundryKeyAuthProvider()`` or any
      ``AuthProvider[str]`` (sent as the ``api-key`` header).
    - **Static Entra token** — pass ``auth=`` an ``AuthProvider`` that returns
      an ``AzureAdToken`` (sent as ``Authorization: Bearer``).
    - **Token provider** — pass ``auth=`` an ``AuthProvider`` that returns a
      ``Callable[[], str]`` (e.g. ``AzureFoundryTokenAuthProvider``); the
      callable is invoked on every request for a fresh bearer token.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth: AuthProvider[AzureFoundryCredential] | None = None,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._auth: AuthProvider[AzureFoundryCredential] = auth or AzureFoundryKeyAuthProvider()
        self._endpoint: str = endpoint
        self._api_version: str = api_version
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
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
        return calculate_azure_foundry_cost(model, usage)

    # MARK: Clients

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            self._sync_client = create_sync_client(
                endpoint=self._endpoint,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = create_async_client(
                endpoint=self._endpoint,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

    @property
    def _query(self) -> dict[str, str]:
        return {"api-version": self._api_version}

    def _sync_headers(self) -> dict[str, str]:
        return auth_headers(self._auth.get_credentials())

    async def _async_headers(self) -> dict[str, str]:
        return auth_headers(await self._auth.aget_credentials())

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
        provider_params: AzureFoundryParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
            response = client.post(
                _chat_path(model), json={**body, "stream": False}, params=self._query, headers=self._sync_headers()
            )
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_chat_completion(parse_body(response, WireCompletion), PROVIDER_NAME, self._calculate_cost)
        return self._apply_multipliers(result, provider_params)

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
        provider_params: AzureFoundryParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
            response = await client.post(
                _chat_path(model),
                json={**body, "stream": False},
                params=self._query,
                headers=await self._async_headers(),
            )
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_chat_completion(parse_body(response, WireCompletion), PROVIDER_NAME, self._calculate_cost)
        return self._apply_multipliers(result, provider_params)

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
        provider_params: AzureFoundryParams | None = None,
    ) -> Iterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
            headers = self._sync_headers()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            with client.stream("POST", _chat_path(model), json=body, params=self._query, headers=headers) as response:
                if response.status_code >= _HTTP_ERROR:
                    response.read()
                    raise error_from_response(response)  # noqa: TRY301
                for _event, data in iter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model, provider_params)
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
        provider_params: AzureFoundryParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
            headers = await self._async_headers()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            async with client.stream(
                "POST", _chat_path(model), json=body, params=self._query, headers=headers
            ) as response:
                if response.status_code >= _HTTP_ERROR:
                    await response.aread()
                    raise error_from_response(response)  # noqa: TRY301
                async for _event, data in aiter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model, provider_params)
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
        provider_params: AzureFoundryParams | None = None,
    ) -> EmbeddingResponse:
        body = self._embed_body(model, input, dimensions, provider_params)
        try:
            client = self._get_sync_client()
            response = client.post(_embeddings_path(model), json=body, params=self._query, headers=self._sync_headers())
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_embedding_response(
            parse_body(response, WireEmbeddingResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_embedding_multipliers(result, provider_params)

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: AzureFoundryParams | None = None,
    ) -> EmbeddingResponse:
        body = self._embed_body(model, input, dimensions, provider_params)
        try:
            client = await self._get_async_client()
            response = await client.post(
                _embeddings_path(model), json=body, params=self._query, headers=await self._async_headers()
            )
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_embedding_response(
            parse_body(response, WireEmbeddingResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_embedding_multipliers(result, provider_params)

    # MARK: Responses

    @override
    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: AzureFoundryParams | None = None,
    ) -> ResponseResponse:
        body = self._responses_body(model, input, provider_params)
        try:
            client = self._get_sync_client()
            response = client.post(_RESPONSES_PATH, json=body, params=self._query, headers=self._sync_headers())
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_responses_response(
            parse_body(response, WireResponsesResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(result, provider_params)

    @override
    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: AzureFoundryParams | None = None,
    ) -> ResponseResponse:
        body = self._responses_body(model, input, provider_params)
        try:
            client = await self._get_async_client()
            response = await client.post(
                _RESPONSES_PATH, json=body, params=self._query, headers=await self._async_headers()
            )
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        result = map_responses_response(
            parse_body(response, WireResponsesResponse), PROVIDER_NAME, self._calculate_cost
        )
        return self._apply_response_multipliers(result, provider_params)

    # MARK: Cost Multipliers

    @staticmethod
    def _cost_multiplier(provider_params: AzureFoundryParams | None) -> float:
        """Compute the combined cost multiplier from provider params."""
        multiplier = 1.0
        if provider_params is None:
            return multiplier
        if provider_params.deployment_type == "data_zone":
            multiplier *= DATA_ZONE_MULTIPLIER
        elif provider_params.deployment_type == "regional":
            multiplier *= REGIONAL_MULTIPLIER
        return multiplier

    @staticmethod
    def _apply_cost_multipliers(cost: Cost | None, provider_params: AzureFoundryParams | None) -> Cost | None:
        if cost is None:
            return None
        multiplier = AzureFoundryProvider._cost_multiplier(provider_params)
        if multiplier == 1.0:
            return cost
        return apply_cost_multiplier(cost, multiplier)

    def _apply_multipliers(self, response: ChatResponse, provider_params: AzureFoundryParams | None) -> ChatResponse:
        """Apply deployment_type cost multipliers to a completed chat response."""
        adjusted = self._apply_cost_multipliers(response.cost, provider_params)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})

    def _apply_embedding_multipliers(
        self, response: EmbeddingResponse, provider_params: AzureFoundryParams | None
    ) -> EmbeddingResponse:
        """Apply deployment_type cost multipliers to an embedding response."""
        adjusted = self._apply_cost_multipliers(response.cost, provider_params)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})

    def _apply_response_multipliers(
        self, response: ResponseResponse, provider_params: AzureFoundryParams | None
    ) -> ResponseResponse:
        """Apply deployment_type cost multipliers to a Responses API response."""
        adjusted = self._apply_cost_multipliers(response.cost, provider_params)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})

    # MARK: Internal Helpers

    def _map_stream_chunk(
        self, chunk: dict[str, Any], model: str, provider_params: AzureFoundryParams | None
    ) -> ChatChunk:
        wire = WireChunk.model_validate(chunk)
        mapped = map_chat_chunk(wire, PROVIDER_NAME)
        if mapped.usage is not None:
            cost = self._calculate_cost(wire.model or model, mapped.usage)
            cost = self._apply_cost_multipliers(cost, provider_params)
            mapped = mapped.model_copy(update={"cost": cost})
        return mapped

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
        provider_params: AzureFoundryParams | None,
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
        provider_params: AzureFoundryParams | None,
    ) -> dict[str, Any]:
        message_dicts = map_messages(messages, explicit_cache=supports_explicit_prompt_cache(model))
        body: dict[str, Any] = {"model": model, "messages": message_dicts}
        if has_cache_breakpoint(message_dicts):
            body["prompt_cache_options"] = {"mode": "explicit"}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            if model.startswith(_MAX_COMPLETION_TOKEN_PREFIXES):
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
            body.update(AzureFoundryProvider._provider_params_kwargs(provider_params))
        return body

    @staticmethod
    def _embed_body(
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: AzureFoundryParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": input}
        if provider_params is not None:
            body.update(AzureFoundryProvider._provider_params_kwargs(provider_params))
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    @staticmethod
    def _responses_body(
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        provider_params: AzureFoundryParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": map_response_input(input), "stream": False}
        body.update(AzureFoundryProvider._responses_kwargs(provider_params))
        return body

    @staticmethod
    def _provider_params_kwargs(params: AzureFoundryParams) -> dict[str, Any]:
        """Convert AzureFoundryParams to request-body kwargs."""
        kwargs: dict[str, Any] = {}
        if params.reasoning_effort is not None:
            kwargs["reasoning_effort"] = params.reasoning_effort
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs

    @staticmethod
    def _responses_kwargs(provider_params: AzureFoundryParams | None) -> dict[str, Any]:
        """Build extra body kwargs for the Responses API."""
        if provider_params is None:
            return {}
        extra: dict[str, Any] = {}
        # Responses API uses reasoning={"effort": ...}, not flat reasoning_effort.
        if provider_params.reasoning_effort is not None:
            extra["reasoning"] = {"effort": provider_params.reasoning_effort}
        if provider_params.seed is not None:
            extra["seed"] = provider_params.seed
        if provider_params.user is not None:
            extra["user"] = provider_params.user
        return extra


def _chat_path(model: str) -> str:
    return f"/deployments/{model}/chat/completions"


def _embeddings_path(model: str) -> str:
    return f"/deployments/{model}/embeddings"
