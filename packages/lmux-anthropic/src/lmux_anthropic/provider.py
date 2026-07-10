"""Anthropic provider implementation."""

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

if TYPE_CHECKING:
    import anthropic
    from google.auth.credentials import Credentials

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, PricingProvider
from lmux.types import ChatChunk, ChatResponse, Cost, Message, ResponseFormat, Tool, ToolChoice, Usage
from lmux_anthropic._exceptions import map_anthropic_error
from lmux_anthropic._lazy import (
    create_async_client,
    create_async_foundry_client,
    create_async_vertex_client,
    create_sync_client,
    create_sync_foundry_client,
    create_sync_vertex_client,
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
from lmux_anthropic.auth import (
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
DEFAULT_MAX_TOKENS = 4096

type SyncAnthropicClient = "anthropic.Anthropic | anthropic.AnthropicVertex | anthropic.AnthropicFoundry"
type AsyncAnthropicClient = (
    "anthropic.AsyncAnthropic | anthropic.AsyncAnthropicVertex | anthropic.AsyncAnthropicFoundry"
)

# Vertex auth providers may return bare credentials, or credentials together
# with the project ID they resolved (e.g. from ADC or a service account file).
type VertexAuthResult = "Credentials | tuple[Credentials, str | None]"

# Foundry auth providers return an API key, or a Microsoft Entra ID bearer
# token provider that the SDK invokes on every request.
type FoundryAuthResult = "str | Callable[[], str]"


def _today() -> date:
    """Return today's date, indirected so tests can pin the pricing clock."""
    return date.today()


class AnthropicProvider(
    CompletionProvider[AnthropicParams],
    PricingProvider,
):
    """Anthropic API provider."""

    _provider_name: ClassVar[str] = PROVIDER_NAME

    def __init__(
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._auth: AuthProvider[str] = auth or AnthropicEnvAuthProvider()
        self._base_url: str | None = base_url
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._default_max_tokens: int = default_max_tokens
        self._sync_client: SyncAnthropicClient | None = None
        self._async_client: AsyncAnthropicClient | None = None
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

    def _create_sync_client(self) -> SyncAnthropicClient:
        return create_sync_client(
            api_key=self._auth.get_credentials(),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    async def _create_async_client(self) -> AsyncAnthropicClient:
        return create_async_client(
            api_key=await self._auth.aget_credentials(),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def _get_sync_client(self) -> SyncAnthropicClient:
        if self._sync_client is None:
            self._sync_client = self._create_sync_client()
        return self._sync_client

    async def _get_async_client(self) -> AsyncAnthropicClient:
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = await self._create_async_client()
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.close()
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
        provider_params: AnthropicParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_chat_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_sync_client()
            message = client.messages.create(**kwargs, stream=False)
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e
        as_of = self._resolve_pricing_as_of(provider_params)
        response = map_message_response(message, self._provider_name, lambda m, u: self._calculate_cost(m, u, as_of))
        return self._apply_multipliers(response, provider_params)

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
        kwargs = self._build_chat_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = await self._get_async_client()
            message = await client.messages.create(**kwargs, stream=False)
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e
        as_of = self._resolve_pricing_as_of(provider_params)
        response = map_message_response(message, self._provider_name, lambda m, u: self._calculate_cost(m, u, as_of))
        return self._apply_multipliers(response, provider_params)

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
        kwargs = self._build_chat_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_sync_client()
            stream = client.messages.create(**kwargs, stream=True)
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e

        as_of = self._resolve_pricing_as_of(provider_params)
        start_usage: Usage | None = None
        start_model: str | None = None
        try:
            for event in stream:
                if event.type == "message_start":
                    start_model, start_usage = map_message_start(event)
                    continue
                if event.type == "content_block_start":
                    chunk = map_content_block_start(event)
                    if chunk is not None:
                        yield chunk
                    continue
                if event.type == "content_block_delta":
                    chunk = map_content_block_delta(event)
                    if chunk is not None:
                        yield chunk
                    continue
                if event.type == "message_delta" and start_usage is not None:
                    chunk = map_message_delta(event, start_usage)
                    cost = self._calculate_cost(model, chunk.usage, as_of) if chunk.usage else None
                    cost = self._apply_cost_multipliers(cost, model, provider_params)
                    yield chunk.model_copy(
                        update={"cost": cost, "model": start_model, "provider": self._provider_name}
                    )
                    continue
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e

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
        kwargs = self._build_chat_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = await self._get_async_client()
            stream = await client.messages.create(**kwargs, stream=True)
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e

        as_of = self._resolve_pricing_as_of(provider_params)
        start_usage: Usage | None = None
        start_model: str | None = None
        try:
            async for event in stream:
                if event.type == "message_start":
                    start_model, start_usage = map_message_start(event)
                    continue
                if event.type == "content_block_start":
                    chunk = map_content_block_start(event)
                    if chunk is not None:
                        yield chunk
                    continue
                if event.type == "content_block_delta":
                    chunk = map_content_block_delta(event)
                    if chunk is not None:
                        yield chunk
                    continue
                if event.type == "message_delta" and start_usage is not None:
                    chunk = map_message_delta(event, start_usage)
                    cost = self._calculate_cost(model, chunk.usage, as_of) if chunk.usage else None
                    cost = self._apply_cost_multipliers(cost, model, provider_params)
                    yield chunk.model_copy(
                        update={"cost": cost, "model": start_model, "provider": self._provider_name}
                    )
                    continue
        except Exception as e:
            raise map_anthropic_error(e, self._provider_name) from e

    # MARK: Internal Helpers

    def _build_chat_kwargs(  # noqa: PLR0913
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
    ) -> dict[str, Any]:
        system, mapped_messages = map_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": mapped_messages,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if tools is not None:
            kwargs["tools"] = map_tools(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = map_tool_choice(tool_choice)
        if response_format is not None:
            output_config = map_response_format(response_format)
            if output_config is not None:
                kwargs["output_config"] = output_config
        # provider_params.thinking takes precedence over reasoning_effort, so skip the
        # reasoning_effort mapping entirely when it is set (otherwise a stray
        # output_config.effort would linger after the provider_params update below).
        provider_sets_thinking = provider_params is not None and provider_params.thinking is not None
        if reasoning_effort is not None and not provider_sets_thinking:
            if model_uses_adaptive_thinking(model):
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {**kwargs.get("output_config", {}), "effort": reasoning_effort}
            else:
                budget = {"low": 1024, "medium": 8192, "high": 32768}[reasoning_effort]
                budget = min(budget, kwargs["max_tokens"] - 1)
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if provider_params is not None:
            kwargs.update(self._provider_params_kwargs(provider_params))
        return kwargs

    @staticmethod
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to kwargs for the Anthropic SDK."""
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

    def _cost_multiplier(self, model: str, provider_params: AnthropicParams | None) -> float:  # pyright: ignore[reportUnusedParameter]
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


class AnthropicVertexProvider(AnthropicProvider):
    """Claude on Vertex AI provider.

    Requires the ``[vertex]`` extra. Reuses the Anthropic API request/response
    handling unchanged; only client creation and auth differ. ``service_tier``
    and ``inference_geo`` are Anthropic-API-only parameters and are dropped
    from outgoing requests; the US-inference cost multiplier never applies.
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

    @staticmethod
    def _split_auth_result(auth_result: "VertexAuthResult") -> "tuple[Credentials, str | None]":
        if isinstance(auth_result, tuple):
            return auth_result
        return auth_result, None

    def _resolve_project_id(self, auth_project_id: str | None) -> str | None:
        """Resolve the project ID: explicit argument, then env var, then auth-derived.

        Matches the SDK's own precedence — the env var wins over a project
        inferred from credentials.
        """
        if self._project_id is not None:
            return self._project_id
        env_project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        if env_project_id:
            return env_project_id
        return auth_project_id

    @override
    def _create_sync_client(self) -> "anthropic.AnthropicVertex":
        credentials, auth_project_id = self._split_auth_result(self._vertex_auth.get_credentials())
        return create_sync_vertex_client(
            credentials=credentials,
            project_id=self._resolve_project_id(auth_project_id),
            region=self._region,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @override
    async def _create_async_client(self) -> "anthropic.AsyncAnthropicVertex":
        credentials, auth_project_id = self._split_auth_result(await self._vertex_auth.aget_credentials())
        return create_async_vertex_client(
            credentials=credentials,
            project_id=self._resolve_project_id(auth_project_id),
            region=self._region,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @staticmethod
    @override
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to SDK kwargs, dropping Anthropic-API-only parameters."""
        kwargs = AnthropicProvider._provider_params_kwargs(params)
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

    Reuses the Anthropic API request/response handling unchanged; only client
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

    @staticmethod
    def _split_foundry_auth(auth_result: "FoundryAuthResult") -> "tuple[str | None, Callable[[], str] | None]":
        if isinstance(auth_result, str):
            return auth_result, None
        return None, auth_result

    @override
    def _create_sync_client(self) -> "anthropic.AnthropicFoundry":
        api_key, azure_ad_token_provider = self._split_foundry_auth(self._foundry_auth.get_credentials())
        return create_sync_foundry_client(
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            resource=self._resource,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @override
    async def _create_async_client(self) -> "anthropic.AsyncAnthropicFoundry":
        api_key, azure_ad_token_provider = self._split_foundry_auth(await self._foundry_auth.aget_credentials())
        return create_async_foundry_client(
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            resource=self._resource,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    @staticmethod
    @override
    def _provider_params_kwargs(params: AnthropicParams) -> dict[str, Any]:
        """Convert AnthropicParams to SDK kwargs, dropping Anthropic-API-only parameters."""
        kwargs = AnthropicProvider._provider_params_kwargs(params)
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
