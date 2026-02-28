"""OpenAI provider implementation."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

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
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._sync_client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_openai_cost(model, usage)

    def _get_sync_client(self) -> "openai.OpenAI":
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

    async def _get_async_client(self) -> "openai.AsyncOpenAI":
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
        provider_params: OpenAIParams | None = None,
    ) -> ChatResponse:
        client = self._get_sync_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            completion = client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_openai_error(e) from e
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
        provider_params: OpenAIParams | None = None,
    ) -> ChatResponse:
        client = await self._get_async_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            completion = await client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_openai_error(e) from e
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
        provider_params: OpenAIParams | None = None,
    ) -> Iterator[ChatChunk]:
        client = self._get_sync_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
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
        provider_params: OpenAIParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        client = await self._get_async_client()
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
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

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: OpenAIParams | None = None,
    ) -> EmbeddingResponse:
        client = self._get_sync_client()
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            response = client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: OpenAIParams | None = None,
    ) -> EmbeddingResponse:
        client = await self._get_async_client()
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            response = await client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)

    # MARK: Responses API

    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        client = self._get_sync_client()
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            response = client.responses.create(model=model, input=map_response_input(input), stream=False, **extra)
        except Exception as e:
            raise map_openai_error(e) from e
        return map_responses_response(response, PROVIDER_NAME, self._calculate_cost)

    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: OpenAIParams | None = None,
    ) -> ResponseResponse:
        client = await self._get_async_client()
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            response = await client.responses.create(
                model=model, input=map_response_input(input), stream=False, **extra
            )
        except Exception as e:
            raise map_openai_error(e) from e
        return map_responses_response(response, PROVIDER_NAME, self._calculate_cost)

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
        if response_format is not None:
            kwargs["response_format"] = map_response_format(response_format)
        if provider_params is not None:
            kwargs.update(OpenAIProvider._provider_params_kwargs(provider_params))
        return kwargs

    @staticmethod
    def _provider_params_kwargs(params: OpenAIParams) -> dict[str, Any]:
        """Convert OpenAIParams to kwargs for the OpenAI SDK."""
        kwargs: dict[str, Any] = {}
        if params.service_tier is not None:
            kwargs["service_tier"] = params.service_tier
        if params.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": params.reasoning_effort}
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs
