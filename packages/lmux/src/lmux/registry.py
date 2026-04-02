"""Prefix-based provider routing registry."""

# ruff: noqa: PLR0913

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Literal

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

    def _build_provider_kwargs(
        self,
        prefix: str,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None,
    ) -> dict[str, Any]:
        """Build kwargs for a provider chat call, omitting reasoning_effort when None."""
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "tools": tools,
            "response_format": response_format,
            "provider_params": self._resolve_params(prefix, provider_params),
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        return kwargs

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ChatResponse:
        """Route a chat completion to the provider registered under *model*'s prefix."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs = self._build_provider_kwargs(
            prefix, temperature, max_tokens, top_p, stop, tools, response_format, reasoning_effort, provider_params
        )
        return provider.chat(bare_model, messages, **kwargs)

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ChatResponse:
        """Async variant of :meth:`chat`."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs = self._build_provider_kwargs(
            prefix, temperature, max_tokens, top_p, stop, tools, response_format, reasoning_effort, provider_params
        )
        return await provider.achat(bare_model, messages, **kwargs)

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> Iterator[ChatChunk]:
        """Streaming variant of :meth:`chat`. Yields :class:`ChatChunk` instances."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs = self._build_provider_kwargs(
            prefix, temperature, max_tokens, top_p, stop, tools, response_format, reasoning_effort, provider_params
        )
        yield from provider.chat_stream(bare_model, messages, **kwargs)

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Async streaming variant of :meth:`chat`. Yields :class:`ChatChunk` instances."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, CompletionProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support chat (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs = self._build_provider_kwargs(
            prefix, temperature, max_tokens, top_p, stop, tools, response_format, reasoning_effort, provider_params
        )
        async for chunk in provider.achat_stream(bare_model, messages, **kwargs):
            yield chunk

    # MARK: Embed

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> EmbeddingResponse:
        """Route an embedding request to the provider registered under *model*'s prefix."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, EmbeddingProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support embeddings (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs: dict[str, Any] = {"provider_params": self._resolve_params(prefix, provider_params)}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return provider.embed(bare_model, input, **kwargs)

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> EmbeddingResponse:
        """Async variant of :meth:`embed`."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, EmbeddingProvider):
            msg = f"Provider {prefix!r} ({type(provider).__name__}) does not support embeddings (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        kwargs: dict[str, Any] = {"provider_params": self._resolve_params(prefix, provider_params)}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return await provider.aembed(bare_model, input, **kwargs)

    # MARK: Responses

    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: BaseProviderParams | Mapping[str, BaseProviderParams] | None = None,
    ) -> ResponseResponse:
        """Route a Responses API request to the provider registered under *model*'s prefix."""
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
        """Async variant of :meth:`create_response`."""
        provider, prefix, bare_model = self._resolve(model)
        if not isinstance(provider, ResponsesProvider):
            provider_name = type(provider).__name__
            msg = f"Provider {prefix!r} ({provider_name}) does not support the Responses API (model: {bare_model!r})"
            raise UnsupportedFeatureError(msg)
        return await provider.acreate_response(
            bare_model, input, provider_params=self._resolve_params(prefix, provider_params)
        )
