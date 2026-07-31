"""Anthropic provider implementation (SDK-lite, httpx transport)."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

    from lmux_bedrock_shared.auth import BedrockAuthContext, BedrockCredentialSource


import base64
from urllib.parse import quote

from lmux._http import aiter_sse, iter_sse
from lmux.cost import ModelPricing, calculate_cost
from lmux.exceptions import LmuxError, ProviderError
from lmux.protocols import AuthProvider, CompletionProvider, PricingProvider
from lmux.types import ChatChunk, ChatResponse, Cost, Message, ResponseFormat, Tool, ToolChoice, Usage
from lmux_anthropic._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_message,
    raise_for_status,
)
from lmux_anthropic._lazy import (
    VERTEX_ANTHROPIC_VERSION,
    bedrock_base_url,
    create_async_bedrock_client,
    create_async_client,
    create_async_foundry_client,
    create_async_vertex_client,
    create_sync_bedrock_client,
    create_sync_client,
    create_sync_foundry_client,
    create_sync_vertex_client,
    foundry_auth_headers,
    vertex_auth_headers,
)
from lmux_anthropic._mappers import (
    map_content_block_delta,
    map_content_block_start,
    map_message_delta,
    map_message_response,
    map_message_start,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
    model_uses_adaptive_thinking,
)
from lmux_anthropic._wire import (
    WireContentBlockDeltaEvent,
    WireContentBlockStartEvent,
    WireMessage,
    WireMessageDeltaEvent,
    WireMessageStartEvent,
)
from lmux_anthropic.auth import (
    AnthropicBedrockEnvAuthProvider,
    AnthropicEnvAuthProvider,
    AnthropicFoundryEnvAuthProvider,
    AnthropicVertexADCAuthProvider,
)
from lmux_anthropic.cost import (
    US_INFERENCE_MULTIPLIER,
    VERTEX_REGIONAL_MULTIPLIER,
    apply_cost_multiplier,
    calculate_anthropic_cost,
    has_vertex_regional_premium,
)
from lmux_anthropic.params import AnthropicParams

PROVIDER_NAME = "anthropic"
VERTEX_PROVIDER_NAME = "anthropic-vertex"
FOUNDRY_PROVIDER_NAME = "anthropic-foundry"
BEDROCK_PROVIDER_NAME = "anthropic-bedrock"
DEFAULT_MAX_TOKENS = 4096
_MESSAGES_PATH = "v1/messages"
_HTTP_ERROR = 400

# AWS Bedrock InvokeModel: the native Anthropic body carries this instead of ``model``/``stream``.
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"
_INVOKE = "invoke"
_INVOKE_STREAM = "invoke-with-response-stream"
_JSON_CONTENT_TYPE = "application/json"
_EXCEPTION_MESSAGE_TYPE = "exception"
_ERROR_MESSAGE_TYPE = "error"

# Vertex auth providers may return bare credentials, or credentials together
# with the project ID they resolved (e.g. from ADC or a service account file).
type VertexAuthResult = "Credentials | tuple[Credentials, str | None]"

# Foundry auth providers return an API key, or a Microsoft Entra ID bearer
# token provider that the client invokes when building the request.
type FoundryAuthResult = "str | Callable[[], str]"


def _today() -> date:
    """Return today's date, indirected so tests can pin the pricing clock."""
    return date.today()


class AnthropicProvider(
    CompletionProvider[AnthropicParams],
    PricingProvider,
):
    """Anthropic API provider over httpx."""

    _provider_name: ClassVar[str] = PROVIDER_NAME

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        transport: "httpx.BaseTransport | None" = None,
        async_transport: "httpx.AsyncBaseTransport | None" = None,
    ) -> None:
        self._auth: AuthProvider[str] = auth or AnthropicEnvAuthProvider()
        self._base_url: str | None = base_url
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._default_max_tokens: int = default_max_tokens
        self._transport: httpx.BaseTransport | None = transport
        self._async_transport: httpx.AsyncBaseTransport | None = async_transport
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    @staticmethod
    def _resolve_pricing_as_of(provider_params: AnthropicParams | None) -> date:
        """Effective pricing date: an explicit ``pricing_as_of`` override, else today."""
        if provider_params is not None and provider_params.pricing_as_of is not None:
            return provider_params.pricing_as_of
        return _today()

    def _calculate_cost(self, model: str, usage: Usage, as_of: date) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing, as_of)
        return calculate_anthropic_cost(model, usage, as_of)

    def _cost_model(self, request_model: str, response_model: str) -> str:  # noqa: ARG002
        """The model ID used to price a response — the resolved model the API reports.

        Bedrock overrides this to the request ID: its pricing table is keyed by region-prefixed
        inference-profile IDs (e.g. ``us.anthropic.claude-opus-4-8``) that the region-less response
        model would not resolve to.
        """
        return response_model

    # MARK: Client Management

    def _create_sync_client(self) -> "httpx.Client":
        return create_sync_client(
            api_key=self._auth.get_credentials(),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
            transport=self._transport,
        )

    async def _create_async_client(self) -> "httpx.AsyncClient":
        return create_async_client(
            api_key=await self._auth.aget_credentials(),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
            transport=self._async_transport,
        )

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            self._sync_client = self._create_sync_client()
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = await self._create_async_client()
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

    # MARK: Request shaping hooks (overridden by Vertex/Foundry)

    def _request_path(self, model: str, *, stream: bool) -> str:  # noqa: ARG002
        """Path for the Messages request; ``model``/``stream`` matter only on Vertex."""
        return _MESSAGES_PATH

    def _transform_body(self, body: dict[str, Any], model: str) -> dict[str, Any]:  # noqa: ARG002
        """Adjust the request body per-transport; identity for the Anthropic API."""
        return body

    def _request_headers(self) -> Mapping[str, str]:
        """Auth headers applied per request (sync path).

        Empty for the direct API — its static ``x-api-key`` lives on the cached client.
        Vertex and Foundry override this to resolve/refresh a short-lived token on every
        request, so a long-lived provider never sends an expired credential.
        """
        return {}

    async def _arequest_headers(self) -> Mapping[str, str]:
        """Auth headers for the async path. Vertex/Foundry override this to offload the
        (blocking) token refresh to a worker thread so it never stalls the event loop.
        """
        return {}

    # MARK: Transport dispatch hooks (overridden by Bedrock for SigV4 + event-stream framing)

    def _post(self, client: "httpx.Client", path: str, body: dict[str, Any]) -> "httpx.Response":
        """Send a non-streaming POST. Bedrock overrides this to SigV4-sign the request body."""
        return client.post(path, json=body, headers=self._request_headers())

    async def _apost(self, client: "httpx.AsyncClient", path: str, body: dict[str, Any]) -> "httpx.Response":
        return await client.post(path, json=body, headers=await self._arequest_headers())

    def _stream_events(self, model: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Open the streaming request and yield raw event dicts (Anthropic SSE by default).

        Bedrock overrides this to sign the request and decode AWS event-stream frames.
        """
        client = self._get_sync_client()
        with client.stream(
            "POST", self._request_path(model, stream=True), json=body, headers=self._request_headers()
        ) as response:
            if response.status_code >= _HTTP_ERROR:
                response.read()
                raise error_from_response(response, self._provider_name)
            for _event, data in iter_sse(response):
                payload = json.loads(data)
                if payload.get("type") == "error":
                    raise error_from_stream(payload, self._provider_name)
                yield payload

    async def _astream_events(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        client = await self._get_async_client()
        async with client.stream(
            "POST", self._request_path(model, stream=True), json=body, headers=await self._arequest_headers()
        ) as response:
            if response.status_code >= _HTTP_ERROR:
                await response.aread()
                raise error_from_response(response, self._provider_name)
            async for _event, data in aiter_sse(response):
                payload = json.loads(data)
                if payload.get("type") == "error":
                    raise error_from_stream(payload, self._provider_name)
                yield payload

    def _raise_for_status(self, response: "httpx.Response") -> None:
        """Raise the mapped error for a non-stream error response. Bedrock overrides for the AWS format."""
        raise_for_status(response, self._provider_name)

    def _map_transport_error(self, error: Exception) -> LmuxError:
        """Map a transport / client-creation failure to an lmux error. Bedrock overrides for botocore creds."""
        return map_transport_error(error, self._provider_name)

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
        provider_params: AnthropicParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params, stream=False,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
            response = self._post(client, self._request_path(model, stream=False), self._transform_body(body, model))
        except Exception as e:
            raise self._map_transport_error(e) from e
        self._raise_for_status(response)
        return self._map_response(model, parse_message(response, self._provider_name), provider_params)

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
        provider_params: AnthropicParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params, stream=False,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
            response = await self._apost(
                client, self._request_path(model, stream=False), self._transform_body(body, model)
            )
        except Exception as e:
            raise self._map_transport_error(e) from e
        self._raise_for_status(response)
        return self._map_response(model, parse_message(response, self._provider_name), provider_params)

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
        provider_params: AnthropicParams | None = None,
    ) -> Iterator[ChatChunk]:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params, stream=True,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        stream = _StreamState(model)
        try:
            for payload in self._stream_events(model, self._transform_body(body, model)):
                chunk = stream.feed(payload)
                if chunk is not None:
                    yield self._finalize_chunk(model, chunk, stream, provider_params, as_of)
        except LmuxError:
            raise
        except Exception as e:
            raise self._map_transport_error(e) from e

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
        provider_params: AnthropicParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params, stream=True,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        stream = _StreamState(model)
        try:
            async for payload in self._astream_events(model, self._transform_body(body, model)):
                chunk = stream.feed(payload)
                if chunk is not None:
                    yield self._finalize_chunk(model, chunk, stream, provider_params, as_of)
        except LmuxError:
            raise
        except Exception as e:
            raise self._map_transport_error(e) from e

    # MARK: Internal Helpers

    def _map_response(self, model: str, message: WireMessage, provider_params: AnthropicParams | None) -> ChatResponse:
        as_of = self._resolve_pricing_as_of(provider_params)
        cost_model = self._cost_model(model, message.model)
        response = map_message_response(
            message, self._provider_name, lambda _m, u: self._calculate_cost(cost_model, u, as_of)
        )
        return self._apply_multipliers(response, provider_params)

    def _finalize_chunk(
        self,
        model: str,
        chunk: ChatChunk,
        stream: "_StreamState",
        provider_params: AnthropicParams | None,
        as_of: date,
    ) -> ChatChunk:
        """Attach model/provider/cost to a message_delta chunk; pass others through."""
        if chunk.usage is None:
            return chunk
        cost = self._calculate_cost(self._cost_model(model, stream.model), chunk.usage, as_of)
        cost = self._apply_cost_multipliers(cost, stream.model, provider_params)
        return chunk.model_copy(update={"cost": cost, "model": stream.model, "provider": self._provider_name})

    def _build_body(  # noqa: PLR0913
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
        provider_params: AnthropicParams | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        system, mapped_messages = map_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "messages": mapped_messages,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "stream": stream,
        }
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if tools is not None:
            body["tools"] = map_tools(tools)
        if tool_choice is not None:
            body["tool_choice"] = map_tool_choice(tool_choice)
        if response_format is not None:
            output_config = map_response_format(response_format)
            if output_config is not None:
                body["output_config"] = output_config
        # provider_params.thinking takes precedence over reasoning_effort, so skip the
        # reasoning_effort mapping entirely when it is set (otherwise a stray
        # output_config.effort would linger after the provider_params update below).
        provider_sets_thinking = provider_params is not None and provider_params.thinking is not None
        if reasoning_effort is not None and not provider_sets_thinking:
            if model_uses_adaptive_thinking(model):
                body["thinking"] = {"type": "adaptive", "display": "summarized"}
                body["output_config"] = {**body.get("output_config", {}), "effort": reasoning_effort}
            else:
                budget = {"low": 1024, "medium": 8192, "high": 32768}[reasoning_effort]
                budget = min(budget, body["max_tokens"] - 1)
                body["thinking"] = {"type": "enabled", "budget_tokens": budget, "display": "summarized"}
        if provider_params is not None:
            body.update(self._provider_params_kwargs(provider_params))
        return body

    @staticmethod
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to request-body kwargs."""
        kwargs: dict[str, Any] = {}
        if params.thinking is not None:
            kwargs["thinking"] = params.thinking
        if params.metadata is not None:
            kwargs["metadata"] = params.metadata
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        if params.service_tier is not None:
            kwargs["service_tier"] = params.service_tier
        if params.inference_geo is not None:
            kwargs["inference_geo"] = params.inference_geo
        if params.cache_control is not None:
            kwargs["cache_control"] = params.cache_control
        return kwargs

    def _cost_multiplier(self, model: str, provider_params: AnthropicParams | None) -> float:  # noqa: ARG002
        """Compute the combined cost multiplier from provider params; ``model`` is a hook for subclasses."""
        multiplier = 1.0
        if provider_params is None:
            return multiplier
        if provider_params.inference_geo == "us":
            multiplier *= US_INFERENCE_MULTIPLIER
        return multiplier

    def _apply_cost_multipliers(
        self, cost: Cost | None, model: str, provider_params: AnthropicParams | None
    ) -> Cost | None:
        if cost is None:
            return None
        multiplier = self._cost_multiplier(model, provider_params)
        if multiplier == 1.0:
            return cost
        return apply_cost_multiplier(cost, multiplier)

    def _apply_multipliers(self, response: ChatResponse, provider_params: AnthropicParams | None) -> ChatResponse:
        """Apply cost multipliers to a completed response."""
        adjusted = self._apply_cost_multipliers(response.cost, response.model, provider_params)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})


class _StreamState:
    """Accumulates message_start context and maps each streamed event to a chunk."""

    def __init__(self, model: str) -> None:
        self.model: str = model
        self._start_usage: Usage | None = None

    def feed(self, event: dict[str, Any]) -> ChatChunk | None:
        """Map one streamed event to a ChatChunk, or None if it carries no delta."""
        event_type = event.get("type")
        if event_type == "message_start":
            self.model, self._start_usage = map_message_start(WireMessageStartEvent.model_validate(event))
            return None
        if event_type == "content_block_start":
            return map_content_block_start(WireContentBlockStartEvent.model_validate(event))
        if event_type == "content_block_delta":
            return map_content_block_delta(WireContentBlockDeltaEvent.model_validate(event))
        if event_type == "message_delta" and self._start_usage is not None:
            return map_message_delta(WireMessageDeltaEvent.model_validate(event), self._start_usage)
        return None


class AnthropicVertexProvider(AnthropicProvider):
    """Claude on Vertex AI provider.

    Requires the ``[vertex]`` extra. Reuses the Anthropic Messages API
    request/response handling; only client creation, endpoint URL, and auth
    differ. ``service_tier`` and ``inference_geo`` are Anthropic-API-only
    parameters and are dropped from outgoing requests; the US-inference cost
    multiplier never applies.
    """

    _provider_name: ClassVar[str] = VERTEX_PROVIDER_NAME

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: AuthProvider["VertexAuthResult"] | None = None,
        project_id: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_max_tokens=default_max_tokens,
        )
        self._vertex_auth: AuthProvider[VertexAuthResult] = auth or AnthropicVertexADCAuthProvider()
        self._project_id: str | None = project_id
        self._region: str | None = region
        self._resolved_project_id: str | None = None
        self._resolved_region: str | None = None
        self._credentials: Credentials | None = None
        self._auth_project_id: str | None = None

    @staticmethod
    def _split_auth_result(auth_result: "VertexAuthResult") -> "tuple[Credentials, str | None]":
        if isinstance(auth_result, tuple):
            return auth_result  # ty: ignore[invalid-return-type]
        return auth_result, None

    def _sync_credentials(self) -> "Credentials":
        """Resolve Google credentials once and cache them; they refresh their own token in place."""
        credentials = self._credentials
        if credentials is None:
            credentials, self._auth_project_id = self._split_auth_result(self._vertex_auth.get_credentials())
            self._credentials = credentials
        return credentials

    async def _acredentials(self) -> "Credentials":
        # Resolved per async-client creation (once per loop); the sync path caches for per-request reuse.
        credentials, self._auth_project_id = self._split_auth_result(await self._vertex_auth.aget_credentials())
        self._credentials = credentials
        return credentials

    def _resolve_project_id(self, auth_project_id: str | None) -> str:
        """Resolve the project ID: explicit argument, then env var, then auth-derived.

        Matches the SDK's own precedence — the env var wins over a project
        inferred from credentials.
        """
        project_id = self._project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or auth_project_id
        if not project_id:
            raise ProviderError(  # noqa: TRY003
                "No project_id was given and it could not be resolved from credentials; pass project_id= "
                "or set ANTHROPIC_VERTEX_PROJECT_ID",
                provider=self._provider_name,
            )
        return project_id

    def _resolve_region(self) -> str:
        region = self._region or os.environ.get("CLOUD_ML_REGION")
        if not region:
            raise ProviderError(  # noqa: TRY003
                "No region was given; pass region= or set CLOUD_ML_REGION", provider=self._provider_name
            )
        return region

    @override
    def _request_headers(self) -> Mapping[str, str]:
        return vertex_auth_headers(self._sync_credentials())

    @override
    async def _arequest_headers(self) -> Mapping[str, str]:
        # The token refresh does blocking HTTP; run it off the event loop.
        return await asyncio.to_thread(vertex_auth_headers, self._sync_credentials())

    @override
    def _create_sync_client(self) -> "httpx.Client":
        self._sync_credentials()
        self._resolved_region = self._resolve_region()
        self._resolved_project_id = self._resolve_project_id(self._auth_project_id)
        return create_sync_vertex_client(
            region=self._resolved_region,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @override
    async def _create_async_client(self) -> "httpx.AsyncClient":
        await self._acredentials()
        self._resolved_region = self._resolve_region()
        self._resolved_project_id = self._resolve_project_id(self._auth_project_id)
        return create_async_vertex_client(
            region=self._resolved_region,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @override
    def _request_path(self, model: str, *, stream: bool) -> str:
        specifier = "streamRawPredict" if stream else "rawPredict"
        return (
            f"projects/{self._resolved_project_id}/locations/{self._resolved_region}"
            f"/publishers/anthropic/models/{model}:{specifier}"
        )

    @override
    def _transform_body(self, body: dict[str, Any], model: str) -> dict[str, Any]:
        transformed = {key: value for key, value in body.items() if key != "model"}
        transformed["anthropic_version"] = VERTEX_ANTHROPIC_VERSION
        return transformed

    @staticmethod
    @override
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to body kwargs, dropping Anthropic-API-only parameters."""
        kwargs = AnthropicProvider._provider_params_kwargs(params)  # noqa: SLF001
        kwargs.pop("service_tier", None)
        kwargs.pop("inference_geo", None)
        return kwargs

    @override
    def _cost_multiplier(self, model: str, provider_params: AnthropicParams | None) -> float:
        """Regional and multi-region Vertex endpoints bill a 10% premium on Claude 4.5+ models.

        The global endpoint bills at list prices. ``inference_geo`` is
        Anthropic-API-only and never contributes a multiplier here.
        """
        region = self._region or os.environ.get("CLOUD_ML_REGION")
        if region == "global":
            return 1.0
        if has_vertex_regional_premium(model):
            return VERTEX_REGIONAL_MULTIPLIER
        return 1.0


class AnthropicFoundryProvider(AnthropicProvider):
    """Claude in Microsoft Foundry provider.

    Reuses the Anthropic Messages API request/response handling; only client
    creation and auth differ. ``service_tier`` and ``inference_geo`` are
    Anthropic-API-only parameters and are dropped from outgoing requests; the
    US-inference cost multiplier never applies.
    """

    _provider_name: ClassVar[str] = FOUNDRY_PROVIDER_NAME

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: AuthProvider["FoundryAuthResult"] | None = None,
        resource: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_max_tokens=default_max_tokens,
        )
        self._foundry_auth: AuthProvider[FoundryAuthResult] = auth or AnthropicFoundryEnvAuthProvider()
        self._resource: str | None = resource
        self._foundry_api_key: str | None = None
        self._foundry_token_provider: Callable[[], str] | None = None
        self._foundry_auth_resolved: bool = False

    @staticmethod
    def _split_foundry_auth(auth_result: "FoundryAuthResult") -> "tuple[str | None, Callable[[], str] | None]":
        if isinstance(auth_result, str):
            return auth_result, None
        return None, auth_result

    def _sync_foundry_auth(self) -> "tuple[str | None, Callable[[], str] | None]":
        """Resolve the Foundry credential once (an API key, or a token-provider callable to invoke per request)."""
        if not self._foundry_auth_resolved:
            self._foundry_api_key, self._foundry_token_provider = self._split_foundry_auth(
                self._foundry_auth.get_credentials()
            )
            self._foundry_auth_resolved = True
        return self._foundry_api_key, self._foundry_token_provider

    async def _afoundry_auth(self) -> "tuple[str | None, Callable[[], str] | None]":
        # Resolved per async-client creation (once per loop); the sync path caches for per-request reuse.
        self._foundry_api_key, self._foundry_token_provider = self._split_foundry_auth(
            await self._foundry_auth.aget_credentials()
        )
        self._foundry_auth_resolved = True
        return self._foundry_api_key, self._foundry_token_provider

    @override
    def _request_headers(self) -> Mapping[str, str]:
        return foundry_auth_headers(*self._sync_foundry_auth())

    @override
    async def _arequest_headers(self) -> Mapping[str, str]:
        # An Entra token provider may do blocking HTTP; run it off the event loop.
        api_key, token_provider = self._sync_foundry_auth()
        return await asyncio.to_thread(foundry_auth_headers, api_key, token_provider)

    @override
    def _create_sync_client(self) -> "httpx.Client":
        self._sync_foundry_auth()
        return create_sync_foundry_client(
            resource=self._resource,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @override
    async def _create_async_client(self) -> "httpx.AsyncClient":
        await self._afoundry_auth()
        return create_async_foundry_client(
            resource=self._resource,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @staticmethod
    @override
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to body kwargs, dropping Anthropic-API-only parameters."""
        kwargs = AnthropicProvider._provider_params_kwargs(params)  # noqa: SLF001
        kwargs.pop("service_tier", None)
        kwargs.pop("inference_geo", None)
        return kwargs

    @override
    def _cost_multiplier(self, model: str, provider_params: AnthropicParams | None) -> float:
        """Foundry bills Anthropic's standard API pricing (Global Standard deployments only).

        ``inference_geo`` is Anthropic-API-only and never contributes a
        multiplier here.
        """
        return 1.0


class AnthropicBedrockProvider(AnthropicProvider):
    """Claude on Amazon Bedrock via the native Anthropic Messages API (InvokeModel).

    Reuses the Anthropic Messages request/response handling; only auth (SigV4, or a Bedrock bearer
    token from ``AWS_BEARER_TOKEN_BEDROCK``), the endpoint path, the request body's
    ``anthropic_version``, AWS event-stream framing, AWS error mapping, and Bedrock pricing differ.
    ``service_tier`` and ``inference_geo`` are Anthropic-API-only parameters and are dropped from
    outgoing requests; the US-inference cost multiplier never applies. Requires the ``[bedrock]``
    extra — its ``lmux-bedrock-shared`` imports are deferred into the methods below (as with the
    Vertex/Foundry providers' optional deps), so importing lmux-anthropic without the extra works.
    Model IDs are the Bedrock forms (``anthropic.claude-opus-4-8`` or an inference-profile ID like
    ``us.anthropic.claude-opus-4-8``).
    """

    _provider_name: ClassVar[str] = BEDROCK_PROVIDER_NAME

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: "BedrockCredentialSource | None" = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        use_fips: bool = False,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        transport: "httpx.BaseTransport | None" = None,
        async_transport: "httpx.AsyncBaseTransport | None" = None,
    ) -> None:
        super().__init__(
            base_url=endpoint_url,
            timeout=timeout,
            max_retries=max_retries,
            default_max_tokens=default_max_tokens,
            transport=transport,
            async_transport=async_transport,
        )
        self._bedrock_auth: BedrockCredentialSource = auth or AnthropicBedrockEnvAuthProvider()
        self._region: str | None = region
        self._use_fips: bool = use_fips
        self._auth_ctx: BedrockAuthContext | None = None

    # MARK: Client & Auth Management

    def _resolve_auth(self) -> "BedrockAuthContext":
        from lmux_bedrock_shared.auth import resolve_auth_context  # noqa: PLC0415

        if self._auth_ctx is None:
            self._auth_ctx = resolve_auth_context(self._bedrock_auth, self._region)
        return self._auth_ctx

    def _bedrock_endpoint(self, auth: "BedrockAuthContext") -> str:
        return self._base_url or bedrock_base_url(auth.region, use_fips=self._use_fips)

    @override
    def _create_sync_client(self) -> "httpx.Client":
        auth = self._resolve_auth()
        return create_sync_bedrock_client(
            base_url=self._bedrock_endpoint(auth),
            timeout=self._timeout,
            max_retries=self._max_retries,
            transport=self._transport,
        )

    @override
    async def _create_async_client(self) -> "httpx.AsyncClient":
        auth = self._resolve_auth()
        return create_async_bedrock_client(
            base_url=self._bedrock_endpoint(auth),
            timeout=self._timeout,
            max_retries=self._max_retries,
            transport=self._async_transport,
        )

    def _build_signed_request(
        self, client: "httpx.Client | httpx.AsyncClient", path: str, body: dict[str, Any]
    ) -> "httpx.Request":
        """Build a POST request with the JSON body and attach Bedrock auth (bearer header or SigV4)."""
        request = client.build_request(
            "POST", path, content=json.dumps(body).encode(), headers={"content-type": _JSON_CONTENT_TYPE}
        )
        self._resolve_auth().apply(request)
        return request

    # MARK: Request shaping

    @override
    def _request_path(self, model: str, *, stream: bool) -> str:
        action = _INVOKE_STREAM if stream else _INVOKE
        return f"/model/{quote(model, safe='')}/{action}"

    @override
    def _transform_body(self, body: dict[str, Any], model: str) -> dict[str, Any]:
        # Bedrock InvokeModel selects the model by URL path and streaming by endpoint; the body
        # carries anthropic_version instead of model/stream.
        transformed = {key: value for key, value in body.items() if key not in ("model", "stream")}
        transformed["anthropic_version"] = _BEDROCK_ANTHROPIC_VERSION
        return transformed

    @staticmethod
    @override
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to body kwargs, dropping Anthropic-API-only parameters."""
        kwargs = AnthropicProvider._provider_params_kwargs(params)  # noqa: SLF001
        kwargs.pop("service_tier", None)
        kwargs.pop("inference_geo", None)
        return kwargs

    # MARK: Cost

    @override
    def _calculate_cost(self, model: str, usage: Usage, as_of: date) -> Cost | None:
        from lmux_bedrock_shared.pricing import calculate_bedrock_anthropic_cost  # noqa: PLC0415

        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing, as_of)
        # Bedrock bills by the Region called, and ``region`` is often left unset and resolved from
        # the session, so price against the resolved auth context rather than the constructor arg.
        return calculate_bedrock_anthropic_cost(model, usage, region=self._resolve_auth().region, as_of=as_of)

    @override
    def _cost_model(self, request_model: str, response_model: str) -> str:
        """Price the Bedrock request ID (with its region profile), not the region-less response ID."""
        return request_model

    @override
    def _cost_multiplier(self, model: str, provider_params: AnthropicParams | None) -> float:
        """Bedrock prices (including regional inference profiles) are exact table entries — no multiplier."""
        return 1.0

    # MARK: Transport dispatch (SigV4-signed requests + AWS event-stream framing)

    @override
    def _post(self, client: "httpx.Client", path: str, body: dict[str, Any]) -> "httpx.Response":
        return client.send(self._build_signed_request(client, path, body))

    @override
    async def _apost(self, client: "httpx.AsyncClient", path: str, body: dict[str, Any]) -> "httpx.Response":
        return await client.send(self._build_signed_request(client, path, body))

    @override
    def _raise_for_status(self, response: "httpx.Response") -> None:
        from lmux_bedrock_shared import exceptions as bedrock_exceptions  # noqa: PLC0415

        bedrock_exceptions.raise_for_status(response, self._provider_name)

    @override
    def _map_transport_error(self, error: Exception) -> LmuxError:
        from lmux_bedrock_shared import exceptions as bedrock_exceptions  # noqa: PLC0415

        return bedrock_exceptions.map_transport_error(error, self._provider_name)

    @override
    def _stream_events(self, model: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        from lmux_bedrock_shared import EventStreamDecoder  # noqa: PLC0415
        from lmux_bedrock_shared import exceptions as bedrock_exceptions  # noqa: PLC0415

        client = self._get_sync_client()
        request = self._build_signed_request(client, self._request_path(model, stream=True), body)
        response = client.send(request, stream=True)
        try:
            if response.is_error:
                response.read()
                bedrock_exceptions.raise_for_status(response, self._provider_name)
            decoder = EventStreamDecoder()
            for raw in response.iter_bytes():
                for headers, payload in decoder.feed(raw):
                    yield self._decode_stream_event(headers, payload)
        finally:
            response.close()

    @override
    async def _astream_events(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        from lmux_bedrock_shared import EventStreamDecoder  # noqa: PLC0415
        from lmux_bedrock_shared import exceptions as bedrock_exceptions  # noqa: PLC0415

        client = await self._get_async_client()
        request = self._build_signed_request(client, self._request_path(model, stream=True), body)
        response = await client.send(request, stream=True)
        try:
            if response.is_error:
                await response.aread()
                bedrock_exceptions.raise_for_status(response, self._provider_name)
            decoder = EventStreamDecoder()
            async for raw in response.aiter_bytes():
                for headers, payload in decoder.feed(raw):
                    yield self._decode_stream_event(headers, payload)
        finally:
            await response.aclose()

    def _decode_stream_event(self, headers: dict[str, str], payload: bytes) -> dict[str, Any]:
        """Decode one Bedrock event-stream frame into an Anthropic event dict; raise on error frames.

        Chunk frames wrap the Anthropic streaming event as base64 under ``bytes`` (the returned dict is
        fed to the shared stream state, which drops non-delta events); a modeled ``exception`` frame
        (``:exception-type``) or unmodeled ``error`` frame (``:error-code``) is raised through the AWS
        error hierarchy.

        A mid-stream Anthropic ``error`` event (e.g. an overload once generation has begun) arrives as
        an ordinary chunk frame on a 200 response, not as an AWS exception frame, so the decoded payload
        is checked for it the same way the SSE transports check their events.
        """
        from lmux_bedrock_shared import exceptions as bedrock_exceptions  # noqa: PLC0415

        data: dict[str, Any] = json.loads(payload) if payload else {}
        message_type = headers.get(":message-type")
        if message_type == _EXCEPTION_MESSAGE_TYPE:
            exception_type = headers.get(":exception-type", "")
            message = data.get("message") or data.get("Message") or exception_type
            raise bedrock_exceptions.error_from_stream_exception(exception_type, str(message), self._provider_name)
        if message_type == _ERROR_MESSAGE_TYPE:
            error_code = headers.get(":error-code", "")
            raise bedrock_exceptions.error_from_stream_exception(
                error_code, headers.get(":error-message") or error_code, self._provider_name
            )
        decoded: dict[str, Any] = json.loads(base64.b64decode(data["bytes"]))
        if decoded.get("type") == "error":
            raise error_from_stream(decoded, self._provider_name)
        return decoded
