"""Prefix-based provider routing registry."""

# ruff: noqa: PLR0913

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from lmux.exceptions import InvalidRequestError, UnsupportedFeatureError
from lmux.protocols import CompletionProvider, EmbeddingProvider, ResponsesProvider
from lmux.types import (
    BaseProviderParams,
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    ResponseInputItem,
    ResponseResponse,
    Tool,
)

type Provider = CompletionProvider[Any] | EmbeddingProvider[Any] | ResponsesProvider[Any]


class Registry:
    """Thin routing layer that maps ``prefix/model`` strings to provider instances.

    Usage::

        registry = Registry()
        registry.register(
            "openai",
            OpenAIProvider(),
            default_params=OpenAIParams(reasoning_effort="high"),
        )
        registry.register("anthropic", AnthropicProvider())
        response = registry.chat("openai/gpt-4o", messages=[...])

    Provider-specific params can be passed per-call as a single ``BaseProviderParams`` or as a
    ``dict`` keyed by prefix for multi-provider loops::

        params = {
            "anthropic": AnthropicParams(inference_geo="us"),
            "openai": OpenAIParams(reasoning_effort="high"),
        }
        for model in ["anthropic/opus-4.6", "openai/gpt-5"]:
            response = registry.chat(model, messages, provider_params=params)
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._default_params: dict[str, BaseProviderParams | None] = {}

    def register[ParamsT: BaseProviderParams | None](
        self,
        prefix: str,
        provider: CompletionProvider[ParamsT] | EmbeddingProvider[ParamsT] | ResponsesProvider[ParamsT],
        *,
        default_params: ParamsT | None = None,
    ) -> None:
        """Register a provider under a prefix (e.g., ``"openai"``, ``"anthropic"``)."""
        self._providers[prefix] = provider
        self._default_params[prefix] = default_params

    def _resolve(self, model: str) -> tuple[Provider, str, str]:
        parts = model.split("/", maxsplit=1)
        if len(parts) != 2:  # noqa: PLR2004
            msg = f"Model string must be in 'prefix/model' format, got: {model!r}"
            raise InvalidRequestError(msg)
        prefix, bare_model = parts
        provider = self._providers.get(prefix)
        if provider is None:
            msg = f"No provider registered for prefix: {prefix!r}"
            raise InvalidRequestError(msg)
        return provider, prefix, bare_model

    def _resolve_params(
        self,
        prefix: str,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None,
    ) -> BaseProviderParams | None:
        """Resolve provider_params to a single BaseProviderParams or None.

        Precedence: per-call params > default_params > None.
        """
        if isinstance(provider_params, Mapping):
            params = provider_params.get(prefix)
        elif isinstance(provider_params, BaseProviderParams):
            params = provider_params
        else:
            params = None
        if params is None:
            params = self._default_params.get(prefix)
        return params

    # MARK: Chat

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
        response_format: ResponseFormat | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ChatResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return provider.chat(
            bare_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            provider_params=self._resolve_params(prefix, provider_params),
        )

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
        response_format: ResponseFormat | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ChatResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return await provider.achat(
            bare_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            provider_params=self._resolve_params(prefix, provider_params),
        )

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
        response_format: ResponseFormat | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> Iterator[ChatChunk]:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        yield from provider.chat_stream(
            bare_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            provider_params=self._resolve_params(prefix, provider_params),
        )

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
        response_format: ResponseFormat | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        async for chunk in provider.achat_stream(
            bare_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            provider_params=self._resolve_params(prefix, provider_params),
        ):
            yield chunk

    # MARK: Embed

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> EmbeddingResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, EmbeddingProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support embeddings (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return provider.embed(bare_model, input, provider_params=self._resolve_params(prefix, provider_params))

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> EmbeddingResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, EmbeddingProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support embeddings (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return await provider.aembed(bare_model, input, provider_params=self._resolve_params(prefix, provider_params))

    # MARK: Responses

    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ResponseResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, ResponsesProvider):
            provider_name = type(provider).__name__
            msg = f"Provider {prefix!r} ({provider_name}) does not support the Responses API (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return provider.create_response(
            bare_model, input, provider_params=self._resolve_params(prefix, provider_params)
        )

    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ResponseResponse:
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, ResponsesProvider):
            provider_name = type(provider).__name__
            msg = f"Provider {prefix!r} ({provider_name}) does not support the Responses API (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return await provider.acreate_response(
            bare_model, input, provider_params=self._resolve_params(prefix, provider_params)
        )
