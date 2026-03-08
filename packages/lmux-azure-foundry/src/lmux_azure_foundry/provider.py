"""Azure AI Foundry provider implementation."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import openai

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, EmbeddingProvider, PricingProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    Tool,
    Usage,
)
from lmux_azure_foundry._exceptions import map_azure_foundry_error
from lmux_azure_foundry._lazy import create_async_client, create_sync_client
from lmux_azure_foundry._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_embedding_response,
    map_messages,
    map_response_format,
    map_tools,
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
DEFAULT_API_VERSION = "2024-12-01-preview"


class AzureFoundryProvider(
    CompletionProvider[AzureFoundryParams],
    EmbeddingProvider[AzureFoundryParams],
    PricingProvider,
):
    """Azure AI Foundry API provider.

    Uses the ``openai`` SDK's ``AzureOpenAI`` / ``AsyncAzureOpenAI`` clients
    to communicate with models deployed in Azure AI Foundry.

    Authentication supports all three methods accepted by the underlying SDK:

    - **API key** — pass ``auth=AzureFoundryKeyAuthProvider()`` or any
      ``AuthProvider[str]``.
    - **Static Azure AD token** — pass ``auth=`` an ``AuthProvider`` that returns
      an ``AzureAdToken``.
    - **Token provider** — pass ``auth=`` an ``AuthProvider`` that returns a
      ``Callable[[], str]`` (e.g. ``AzureFoundryTokenAuthProvider``).
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
        self._endpoint = endpoint
        self._api_version = api_version
        self._timeout = timeout
        self._max_retries = max_retries
        self._sync_client: openai.AzureOpenAI | None = None
        self._async_client: openai.AsyncAzureOpenAI | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_azure_foundry_cost(model, usage)

    def _get_sync_client(self) -> "openai.AzureOpenAI":
        if self._sync_client is None:
            credential = self._auth.get_credentials()
            self._sync_client = create_sync_client(
                credential=credential,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sync_client

    async def _get_async_client(self) -> "openai.AsyncAzureOpenAI":
        if self._async_client is None:
            credential = await self._auth.aget_credentials()
            self._async_client = create_async_client(
                credential=credential,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
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
        provider_params: AzureFoundryParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = self._get_sync_client()
            completion = client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_azure_foundry_error(e) from e
        response = map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)
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
        provider_params: AzureFoundryParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            client = await self._get_async_client()
            completion = await client.chat.completions.create(**kwargs, stream=False)
        except Exception as e:
            raise map_azure_foundry_error(e) from e
        response = map_chat_completion(completion, PROVIDER_NAME, self._calculate_cost)
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
        provider_params: AzureFoundryParams | None = None,
    ) -> Iterator[ChatChunk]:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
            client = self._get_sync_client()
            stream = client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_azure_foundry_error(e) from e

        try:
            for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    cost = self._calculate_cost(chunk.model, mapped.usage)
                    cost = self._apply_cost_multipliers(cost, provider_params)
                    mapped = mapped.model_copy(update={"cost": cost})
                yield mapped
        except Exception as e:
            raise map_azure_foundry_error(e) from e

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
        provider_params: AzureFoundryParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        kwargs = self._build_chat_kwargs(
            model, messages, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        kwargs["stream_options"] = {"include_usage": True}
        try:
            client = await self._get_async_client()
            stream = await client.chat.completions.create(**kwargs, stream=True)
        except Exception as e:
            raise map_azure_foundry_error(e) from e

        try:
            async for chunk in stream:
                mapped = map_chat_chunk(chunk)
                if mapped.usage is not None:
                    cost = self._calculate_cost(chunk.model, mapped.usage)
                    cost = self._apply_cost_multipliers(cost, provider_params)
                    mapped = mapped.model_copy(update={"cost": cost})
                yield mapped
        except Exception as e:
            raise map_azure_foundry_error(e) from e

    # MARK: Embeddings

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: AzureFoundryParams | None = None,
    ) -> EmbeddingResponse:
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            client = self._get_sync_client()
            response = client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_azure_foundry_error(e) from e
        result = map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)
        return self._apply_embedding_multipliers(result, provider_params)

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: AzureFoundryParams | None = None,
    ) -> EmbeddingResponse:
        extra: dict[str, Any] = self._provider_params_kwargs(provider_params) if provider_params else {}
        try:
            client = await self._get_async_client()
            response = await client.embeddings.create(model=model, input=input, **extra)
        except Exception as e:
            raise map_azure_foundry_error(e) from e
        result = map_embedding_response(response, PROVIDER_NAME, self._calculate_cost)
        return self._apply_embedding_multipliers(result, provider_params)

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
        provider_params: AzureFoundryParams | None,
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
            kwargs.update(AzureFoundryProvider._provider_params_kwargs(provider_params))
        return kwargs

    @staticmethod
    def _provider_params_kwargs(params: AzureFoundryParams) -> dict[str, Any]:
        """Convert AzureFoundryParams to kwargs for the OpenAI SDK."""
        kwargs: dict[str, Any] = {}
        if params.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": params.reasoning_effort}
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs
