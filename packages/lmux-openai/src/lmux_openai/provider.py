"""OpenAI provider implementation."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    import openai

from lmux.cost import ModelPricing, calculate_cost
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
from lmux_openai._exceptions import map_openai_error
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
from lmux_openai.auth import OpenAIEnvAuthProvider
from lmux_openai.cost import calculate_openai_cost
from lmux_openai.params import OpenAIParams

PROVIDER_NAME = "openai"


class OpenAIProvider(
    CompletionProvider[OpenAIParams],
    EmbeddingProvider[OpenAIParams],
    ResponsesProvider[OpenAIParams],
    PricingProvider,
):
    """OpenAI API provider."""

    def __init__(
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._auth: AuthProvider[str] = auth or OpenAIEnvAuthProvider()
        self._base_url: str | None = base_url
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._sync_client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None
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

    def _get_sync_client(self) -> "openai.OpenAI":
        if self._sync_client is None:
            self._sync_client = create_sync_client(
                api_key=self._auth.get_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sync_client

    async def _get_async_client(self) -> "openai.AsyncOpenAI":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = create_async_client(
                api_key=await self._auth.aget_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
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
        provider_params: OpenAIParams | None = None,
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
            completion = client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)

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
            completion = await client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)

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
        kwargs["stream_options"] = {"include_usage": True}
        try:
            client = self._get_sync_client()
            stream = client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_openai_error(e) from e

        try:
            for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(chunk.model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_openai_error(e) from e

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
        kwargs["stream_options"] = {"include_usage": True}
        try:
            client = await self._get_async_client()
            stream = await client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_openai_error(e) from e

        try:
            async for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(chunk.model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_openai_error(e) from e

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
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        if dimensions is not None:
            extra["dimensions"] = dimensions
        try:
            client = self._get_sync_client()
            response = client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: OpenAIParams | None = None,
    ) -> EmbeddingResponse:
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        if dimensions is not None:
            extra["dimensions"] = dimensions
        try:
            client = await self._get_async_client()
            response = await client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)

    # MARK: Responses API

    @override
    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        extra = self._responses_kwargs(provider_params)
        try:
            client = self._get_sync_client()
            response = client.responses.create(model=model, input=map_response_input(input), stream=False, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_responses_response(response, PROVIDER_NAME, self._calculate_cost)

    @override
    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        extra = self._responses_kwargs(provider_params)
        try:
            client = await self._get_async_client()
            response = await client.responses.create(
                model=model, input=map_response_input(input), stream=False, **extra
            )
        except Exception as e:
            raise map_openai_error(e) from e
        return map_responses_response(response, PROVIDER_NAME, self._calculate_cost)

    # MARK: Internal Helpers

    @staticmethod
    def _responses_kwargs(provider_params: OpenAIParams | None) -> dict[str, Any]:
        """Build extra kwargs for the Responses API."""
        extra: dict[str, Any] = OpenAIProvider._provider_params_kwargs(provider_params) if provider_params else {}
        # Responses API uses reasoning={"effort": ...}, not flat reasoning_effort
        if provider_params is not None and provider_params.reasoning_effort is not None:
            extra["reasoning"] = {"effort": provider_params.reasoning_effort}
        return extra

    @staticmethod
    def _build_chat_kwargs(  # noqa: PLR0913
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
        if tool_choice is not None:
            kwargs["tool_choice"] = map_tool_choice(tool_choice)
        if response_format is not None:
            kwargs["response_format"] = map_response_format(response_format)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if provider_params is not None:
            kwargs.update(OpenAIProvider._provider_params_kwargs(provider_params))
            # Chat Completions uses flat reasoning_effort; provider_params overrides top-level
            if provider_params.reasoning_effort is not None:
                kwargs["reasoning_effort"] = provider_params.reasoning_effort
        return kwargs

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
