"""Tests for lmux MockProvider."""

import pytest

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import LmuxError, ProviderError
from lmux.mock import MockProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    ResponseResponse,
    Usage,
    UserMessage,
)

SAMPLE_USAGE = Usage(input_tokens=10, output_tokens=5)
SAMPLE_COST = Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03)


@pytest.fixture
def chat_response() -> ChatResponse:
    return ChatResponse(
        content="Hello!",
        usage=SAMPLE_USAGE,
        cost=SAMPLE_COST,
        model="test-model",
        provider="mock",
    )


@pytest.fixture
def embed_response() -> EmbeddingResponse:
    return EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        usage=Usage(input_tokens=5, output_tokens=0),
        cost=None,
        model="test-embed",
        provider="mock",
    )


@pytest.fixture
def response_response() -> ResponseResponse:
    return ResponseResponse(
        id="resp_123",
        output_text="Hi!",
        usage=SAMPLE_USAGE,
        cost=None,
        model="test-model",
        provider="mock",
    )


@pytest.fixture
def chat_chunks() -> list[ChatChunk]:
    return [
        ChatChunk(delta="Hel"),
        ChatChunk(delta="lo!"),
        ChatChunk(usage=SAMPLE_USAGE, finish_reason="stop"),
    ]


class TestMockChat:
    def test_returns_configured_response(self, chat_response: ChatResponse) -> None:
        provider = MockProvider(chat_responses=[chat_response])
        result = provider.chat("test-model", [UserMessage(content="Hi")])
        assert result == chat_response

    def test_cycles_through_responses(self) -> None:
        r1 = ChatResponse(content="First", usage=SAMPLE_USAGE, cost=SAMPLE_COST, model="m", provider="mock")
        r2 = ChatResponse(content="Second", usage=SAMPLE_USAGE, cost=SAMPLE_COST, model="m", provider="mock")
        provider = MockProvider(chat_responses=[r1, r2])
        assert provider.chat("m", [UserMessage(content="Hi")]).content == "First"
        assert provider.chat("m", [UserMessage(content="Hi")]).content == "Second"
        assert provider.chat("m", [UserMessage(content="Hi")]).content == "First"

    def test_records_calls(self, chat_response: ChatResponse) -> None:
        provider = MockProvider(chat_responses=[chat_response])
        messages = [UserMessage(content="Hi")]
        _ = provider.chat("gpt-4o", messages)
        assert len(provider.calls) == 1
        assert provider.calls[0].method == "chat"
        assert provider.calls[0].model == "gpt-4o"
        assert provider.calls[0].messages == messages

    def test_raises_error_when_no_responses(self) -> None:
        provider = MockProvider()
        with pytest.raises(IndexError, match="No chat responses configured"):
            _ = provider.chat("m", [UserMessage(content="Hi")])


class TestMockAchat:
    async def test_returns_configured_response(self, chat_response: ChatResponse) -> None:
        provider = MockProvider(chat_responses=[chat_response])
        result = await provider.achat("test-model", [UserMessage(content="Hi")])
        assert result == chat_response

    async def test_records_calls(self, chat_response: ChatResponse) -> None:
        provider = MockProvider(chat_responses=[chat_response])
        _ = await provider.achat("m", [UserMessage(content="Hi")])
        assert provider.calls[0].method == "achat"


class TestMockChatStream:
    def test_yields_chunks(self, chat_chunks: list[ChatChunk]) -> None:
        provider = MockProvider(chat_stream_responses=[chat_chunks])
        result = list(provider.chat_stream("m", [UserMessage(content="Hi")]))
        assert result == chat_chunks

    def test_records_calls(self, chat_chunks: list[ChatChunk]) -> None:
        provider = MockProvider(chat_stream_responses=[chat_chunks])
        _ = list(provider.chat_stream("m", [UserMessage(content="Hi")]))
        assert provider.calls[0].method == "chat_stream"

    def test_raises_error_when_no_responses(self) -> None:
        provider = MockProvider()
        with pytest.raises(IndexError, match="No chat stream responses configured"):
            _ = list(provider.chat_stream("m", [UserMessage(content="Hi")]))


class TestMockAchatStream:
    async def test_yields_chunks(self, chat_chunks: list[ChatChunk]) -> None:
        provider = MockProvider(chat_stream_responses=[chat_chunks])
        result = [chunk async for chunk in provider.achat_stream("m", [UserMessage(content="Hi")])]
        assert result == chat_chunks

    async def test_records_calls(self, chat_chunks: list[ChatChunk]) -> None:
        provider = MockProvider(chat_stream_responses=[chat_chunks])
        _ = [chunk async for chunk in provider.achat_stream("m", [UserMessage(content="Hi")])]
        assert provider.calls[0].method == "achat_stream"


class TestMockEmbed:
    def test_returns_configured_response(self, embed_response: EmbeddingResponse) -> None:
        provider = MockProvider(embed_responses=[embed_response])
        result = provider.embed("test-embed", "hello")
        assert result == embed_response

    def test_records_calls(self, embed_response: EmbeddingResponse) -> None:
        provider = MockProvider(embed_responses=[embed_response])
        _ = provider.embed("test-embed", ["hello", "world"])
        assert provider.calls[0].method == "embed"
        assert provider.calls[0].text == ["hello", "world"]

    def test_raises_error_when_no_responses(self) -> None:
        provider = MockProvider()
        with pytest.raises(IndexError, match="No embed responses configured"):
            _ = provider.embed("m", "hello")

    async def test_aembed(self, embed_response: EmbeddingResponse) -> None:
        provider = MockProvider(embed_responses=[embed_response])
        result = await provider.aembed("test-embed", "hello")
        assert result == embed_response
        assert provider.calls[0].method == "aembed"


class TestMockCreateResponse:
    def test_returns_configured_response(self, response_response: ResponseResponse) -> None:
        provider = MockProvider(response_responses=[response_response])
        result = provider.create_response("test-model", "Hi")
        assert result == response_response

    def test_records_calls(self, response_response: ResponseResponse) -> None:
        provider = MockProvider(response_responses=[response_response])
        _ = provider.create_response("test-model", "Hi")
        assert provider.calls[0].method == "create_response"
        assert provider.calls[0].input_data == "Hi"

    def test_raises_error_when_no_responses(self) -> None:
        provider = MockProvider()
        with pytest.raises(IndexError, match="No response responses configured"):
            _ = provider.create_response("m", "Hi")

    async def test_acreate_response(self, response_response: ResponseResponse) -> None:
        provider = MockProvider(response_responses=[response_response])
        result = await provider.acreate_response("test-model", "Hi")
        assert result == response_response
        assert provider.calls[0].method == "acreate_response"


class TestMockErrors:
    def test_raises_configured_errors(self, chat_response: ChatResponse) -> None:
        error = ProviderError("boom", provider="mock")
        provider = MockProvider(chat_responses=[chat_response], errors=[error])
        with pytest.raises(ProviderError, match="boom"):
            _ = provider.chat("m", [UserMessage(content="Hi")])

    def test_error_then_success(self, chat_response: ChatResponse) -> None:
        error = ProviderError("boom", provider="mock")
        provider = MockProvider(chat_responses=[chat_response], errors=[error])
        with pytest.raises(ProviderError):
            _ = provider.chat("m", [UserMessage(content="Hi")])
        # Second call succeeds because errors are exhausted
        result = provider.chat("m", [UserMessage(content="Hi")])
        assert result == chat_response

    def test_errors_across_methods(self, chat_response: ChatResponse, embed_response: EmbeddingResponse) -> None:
        error = ProviderError("boom", provider="mock")
        provider = MockProvider(
            chat_responses=[chat_response],
            embed_responses=[embed_response],
            errors=[error],
        )
        with pytest.raises(ProviderError):
            _ = provider.embed("m", "hello")
        # Next call to any method succeeds
        result = provider.chat("m", [UserMessage(content="Hi")])
        assert result == chat_response

    async def test_async_errors(self, chat_response: ChatResponse) -> None:
        error = ProviderError("boom", provider="mock")
        provider = MockProvider(chat_responses=[chat_response], errors=[error])
        with pytest.raises(LmuxError):
            _ = await provider.achat("m", [UserMessage(content="Hi")])


class TestRegisterPricing:
    def test_register_pricing_stores_model(self) -> None:
        provider = MockProvider()
        pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.001, output_cost_per_token=0.002)],
        )
        provider.register_pricing("my-model", pricing)
        assert provider._custom_pricing["my-model"] == pricing  # pyright: ignore[reportPrivateUsage]

    def test_register_pricing_overwrites(self) -> None:
        provider = MockProvider()
        first = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.001, output_cost_per_token=0.002)],
        )
        second = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.003, output_cost_per_token=0.004)],
        )
        provider.register_pricing("my-model", first)
        provider.register_pricing("my-model", second)
        assert provider._custom_pricing["my-model"] == second  # pyright: ignore[reportPrivateUsage]
