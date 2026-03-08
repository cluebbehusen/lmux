"""Anthropic provider implementation."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, PricingProvider
from lmux.types import ChatChunk, ChatResponse, Cost, Message, ResponseFormat, Tool, Usage
from lmux_anthropic._exceptions import map_anthropic_error
from lmux_anthropic._lazy import create_async_client, create_sync_client
from lmux_anthropic._mappers import (
    map_content_block_delta,
    map_content_block_start,
    map_message_delta,
    map_message_response,
    map_message_start,
    map_messages,
    map_response_format,
    map_tools,
)
from lmux_anthropic.auth import AnthropicEnvAuthProvider
from lmux_anthropic.cost import (
    US_INFERENCE_MULTIPLIER,
    apply_cost_multiplier,
    calculate_anthropic_cost,
)
from lmux_anthropic.params import AnthropicParams

PROVIDER_NAME = "anthropic"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(
    CompletionProvider[AnthropicParams],
    PricingProvider,
):
    """Anthropic API provider."""

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
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_max_tokens = default_max_tokens
        self._sync_client: anthropic.Anthropic | None = None
        self._async_client: anthropic.AsyncAnthropic | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_anthropic_cost(model, usage)

    def _get_sync_client(self) -> "anthropic.Anthropic":
        if self._sync_client is None:
            self._sync_client = create_sync_client(
                api_key=self._auth.get_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sync_client

    async def _get_async_client(self) -> "anthropic.AsyncAnthropic":
        if self._async_client is None:
            self._async_client = create_async_client(
                api_key=await self._auth.aget_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._async_client

    # MARK: Chat

    def chat(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        provider_params: AnthropicParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = self._get_sync_client()
            message = client.messages.create(**kwargs, stream=False)
        except Exception as e:
            raise map_anthropic_error(e) from e
        response = map_message_response(message, PROVIDER_NAME, self._calculate_cost)
        return self._apply_multipliers(response, provider_params)

    async def achat(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        provider_params: AnthropicParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = await self._get_async_client()
            message = await client.messages.create(**kwargs, stream=False)
        except Exception as e:
            raise map_anthropic_error(e) from e
        response = map_message_response(message, PROVIDER_NAME, self._calculate_cost)
        return self._apply_multipliers(response, provider_params)

    def chat_stream(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        provider_params: AnthropicParams | None = None,
    ) -> Iterator[ChatChunk]:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = self._get_sync_client()
            stream = client.messages.create(**kwargs, stream=True)
        except Exception as e:
            raise map_anthropic_error(e) from e

        start_usage: Usage | None = None
        try:
            for event in stream:
                if event.type == "message_start":
                    start_usage = map_message_start(event)
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
                    cost = self._calculate_cost(model, chunk.usage) if chunk.usage else None
                    cost = self._apply_cost_multipliers(cost, provider_params)
                    yield chunk.model_copy(update={"cost": cost})
                    continue
        except Exception as e:
            raise map_anthropic_error(e) from e

    async def achat_stream(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        provider_params: AnthropicParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = await self._get_async_client()
            stream = await client.messages.create(**kwargs, stream=True)
        except Exception as e:
            raise map_anthropic_error(e) from e

        start_usage: Usage | None = None
        try:
            async for event in stream:
                if event.type == "message_start":
                    start_usage = map_message_start(event)
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
                    cost = self._calculate_cost(model, chunk.usage) if chunk.usage else None
                    cost = self._apply_cost_multipliers(cost, provider_params)
                    yield chunk.model_copy(update={"cost": cost})
                    continue
        except Exception as e:
            raise map_anthropic_error(e) from e

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
        response_format: ResponseFormat | None,
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
        if response_format is not None:
            output_config = map_response_format(response_format)
            if output_config is not None:
                kwargs["output_config"] = output_config
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
        return kwargs

    @staticmethod
    def _cost_multiplier(provider_params: AnthropicParams | None) -> float:
        """Compute the combined cost multiplier from provider params."""
        multiplier = 1.0
        if provider_params is None:
            return multiplier
        if provider_params.inference_geo == "us":
            multiplier *= US_INFERENCE_MULTIPLIER
        return multiplier

    @staticmethod
    def _apply_cost_multipliers(cost: Cost | None, provider_params: AnthropicParams | None) -> Cost | None:
        if cost is None:
            return None
        multiplier = AnthropicProvider._cost_multiplier(provider_params)
        if multiplier == 1.0:
            return cost
        return apply_cost_multiplier(cost, multiplier)

    def _apply_multipliers(self, response: ChatResponse, provider_params: AnthropicParams | None) -> ChatResponse:
        """Apply inference_geo cost multipliers to a completed response."""
        adjusted = self._apply_cost_multipliers(response.cost, provider_params)
        if adjusted is response.cost:
            return response
        return response.model_copy(update={"cost": adjusted})
