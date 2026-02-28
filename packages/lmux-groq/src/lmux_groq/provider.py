"""Groq provider implementation."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import groq

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, PricingProvider
from lmux.types import ChatChunk, ChatResponse, Cost, Message, ResponseFormat, Tool, Usage
from lmux_groq._exceptions import map_groq_error
from lmux_groq._lazy import create_async_client, create_sync_client
from lmux_groq._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_messages,
    map_response_format,
    map_tools,
)
from lmux_groq.auth import GroqEnvAuthProvider
from lmux_groq.cost import calculate_groq_cost
from lmux_groq.params import GroqParams

PROVIDER_NAME = "groq"


class GroqProvider(
    CompletionProvider[GroqParams],
    PricingProvider,
):
    """Groq API provider."""

    def __init__(
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._auth: AuthProvider[str] = auth or GroqEnvAuthProvider()
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._sync_client: groq.Groq | None = None
        self._async_client: groq.AsyncGroq | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_groq_cost(model, usage)

    def _get_sync_client(self) -> "groq.Groq":
        if self._sync_client is None:
            api_key = self._auth.get_credentials()
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            if self._max_retries is not None:
                kwargs["max_retries"] = self._max_retries
            self._sync_client = create_sync_client(**kwargs)
        return self._sync_client

    async def _get_async_client(self) -> "groq.AsyncGroq":
        if self._async_client is None:
            api_key = await self._auth.aget_credentials()
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            if self._max_retries is not None:
                kwargs["max_retries"] = self._max_retries
            self._async_client = create_async_client(**kwargs)
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
        provider_params: GroqParams | None = None,
    ) -> ChatResponse:
        client = self._get_sync_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            completion = client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_groq_error(e) from e
        return map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GroqParams | None = None,
    ) -> ChatResponse:
        client = await self._get_async_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            completion = await client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_groq_error(e) from e
        return map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GroqParams | None = None,
    ) -> Iterator[ChatChunk]:
        client = self._get_sync_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_groq_error(e) from e

        try:
            for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(chunk.model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_groq_error(e) from e

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
        provider_params: GroqParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        client = await self._get_async_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_groq_error(e) from e

        try:
            async for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(chunk.model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_groq_error(e) from e

    # MARK: Internal Helpers

    @staticmethod
    def _build_chat_kwargs(  # noqa: PLR0913
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        response_format: ResponseFormat | None,
        provider_params: GroqParams | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": map_messages(messages),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = stop
        if tools is not None:
            kwargs["tools"] = map_tools(tools)
        if response_format is not None:
            kwargs["response_format"] = map_response_format(response_format)
        if provider_params is not None:
            kwargs.update(GroqProvider._provider_params_kwargs(provider_params))
        return kwargs

    @staticmethod
    def _provider_params_kwargs(params: GroqParams) -> dict[str, Any]:
        """Convert GroqParams to kwargs for the Groq SDK."""
        kwargs: dict[str, Any] = {}
        if params.service_tier is not None:
            kwargs["service_tier"] = params.service_tier
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs
