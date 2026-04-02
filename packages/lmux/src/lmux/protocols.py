"""Provider protocols defining the interface contracts for lmux providers."""

# ruff: noqa: PLR0913

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from lmux.cost import ModelPricing
from lmux.types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    ResponseInputItem,
    ResponseResponse,
    Tool,
)


@runtime_checkable
class AuthProvider[AuthT, AAuthT = AuthT](Protocol):
    """Protocol for providing authentication credentials to a provider."""

    def get_credentials(self) -> AuthT: ...
    async def aget_credentials(self) -> AAuthT: ...


@runtime_checkable
class CompletionProvider[ParamsT](Protocol):
    """Protocol for providers that support chat completions."""

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
        provider_params: ParamsT | None = None,
    ) -> ChatResponse: ...

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
        provider_params: ParamsT | None = None,
    ) -> ChatResponse: ...

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
        provider_params: ParamsT | None = None,
    ) -> Iterator[ChatChunk]: ...

    def achat_stream(
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
        provider_params: ParamsT | None = None,
    ) -> AsyncIterator[ChatChunk]: ...


@runtime_checkable
class EmbeddingProvider[ParamsT](Protocol):
    """Protocol for providers that support embeddings."""

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: ParamsT | None = None,
    ) -> EmbeddingResponse: ...

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: ParamsT | None = None,
    ) -> EmbeddingResponse: ...


@runtime_checkable
class ResponsesProvider[ParamsT](Protocol):
    """Protocol for providers that support the Responses API style."""

    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: ParamsT | None = None,
    ) -> ResponseResponse: ...

    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],  # noqa: A002
        *,
        provider_params: ParamsT | None = None,
    ) -> ResponseResponse: ...


@runtime_checkable
class PricingProvider(Protocol):
    """Protocol for providers that support custom model pricing registration."""

    def register_pricing(self, model: str, pricing: ModelPricing) -> None: ...
