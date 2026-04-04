"""Mock provider for testing lmux consumer code."""

# pyright: reportUnusedParameter=false

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, override

from lmux.cost import ModelPricing
from lmux.exceptions import LmuxError
from lmux.protocols import CompletionProvider, EmbeddingProvider, PricingProvider, ResponsesProvider
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


@dataclass
class MockCallRecord:
    """Record of a call made to the mock provider."""

    method: str
    model: str
    messages: Sequence[Message] | None = None
    text: str | list[str] | None = None
    input_data: str | Sequence[ResponseInputItem] | None = None


class MockProvider(
    CompletionProvider[None],
    EmbeddingProvider[None],
    ResponsesProvider[None],
    PricingProvider,
):
    """A configurable mock provider for testing.

    Cycles through preconfigured responses and tracks all calls made.
    """

    def __init__(
        self,
        *,
        chat_responses: list[ChatResponse] | None = None,
        chat_stream_responses: list[list[ChatChunk]] | None = None,
        embed_responses: list[EmbeddingResponse] | None = None,
        response_responses: list[ResponseResponse] | None = None,
        errors: list[LmuxError] | None = None,
    ) -> None:
        self._chat_responses: list[ChatResponse] = chat_responses or []
        self._chat_stream_responses: list[list[ChatChunk]] = chat_stream_responses or []
        self._embed_responses: list[EmbeddingResponse] = embed_responses or []
        self._response_responses: list[ResponseResponse] = response_responses or []
        self._errors: list[LmuxError] = errors or []
        self._error_index: int = 0
        self._chat_index: int = 0
        self._chat_stream_index: int = 0
        self._embed_index: int = 0
        self._response_index: int = 0
        self._custom_pricing: dict[str, ModelPricing] = {}
        self.calls: list[MockCallRecord] = []

    # MARK: PricingProvider

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _maybe_raise(self) -> None:
        if self._error_index < len(self._errors):
            error = self._errors[self._error_index]
            self._error_index += 1
            raise error

    def _next_chat(self) -> ChatResponse:
        if not self._chat_responses:
            msg = "No chat responses configured"
            raise IndexError(msg)
        response = self._chat_responses[self._chat_index % len(self._chat_responses)]
        self._chat_index += 1
        return response

    def _next_chat_stream(self) -> list[ChatChunk]:
        if not self._chat_stream_responses:
            msg = "No chat stream responses configured"
            raise IndexError(msg)
        response = self._chat_stream_responses[self._chat_stream_index % len(self._chat_stream_responses)]
        self._chat_stream_index += 1
        return response

    def _next_embed(self) -> EmbeddingResponse:
        if not self._embed_responses:
            msg = "No embed responses configured"
            raise IndexError(msg)
        response = self._embed_responses[self._embed_index % len(self._embed_responses)]
        self._embed_index += 1
        return response

    def _next_response(self) -> ResponseResponse:
        if not self._response_responses:
            msg = "No response responses configured"
            raise IndexError(msg)
        response = self._response_responses[self._response_index % len(self._response_responses)]
        self._response_index += 1
        return response

    # MARK: CompletionProvider[None]

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> ChatResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="chat", model=model, messages=messages))
        return self._next_chat()

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> ChatResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="achat", model=model, messages=messages))
        return self._next_chat()

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> Iterator[ChatChunk]:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="chat_stream", model=model, messages=messages))
        yield from self._next_chat_stream()

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> AsyncIterator[ChatChunk]:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="achat_stream", model=model, messages=messages))
        for chunk in self._next_chat_stream():
            yield chunk

    # MARK: EmbeddingProvider[None]

    @override
    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: None = None,
    ) -> EmbeddingResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="embed", model=model, text=input))
        return self._next_embed()

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: None = None,
    ) -> EmbeddingResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="aembed", model=model, text=input))
        return self._next_embed()

    # MARK: ResponsesProvider[None]

    @override
    def create_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: None = None,
    ) -> ResponseResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="create_response", model=model, input_data=input))
        return self._next_response()

    @override
    async def acreate_response(
        self,
        model: str,
        input: str | Sequence[ResponseInputItem],
        *,
        provider_params: None = None,
    ) -> ResponseResponse:
        self._maybe_raise()
        self.calls.append(MockCallRecord(method="acreate_response", model=model, input_data=input))
        return self._next_response()
