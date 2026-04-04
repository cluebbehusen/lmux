"""Tests for the prefix-based routing registry."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from lmux.exceptions import InvalidRequestError, UnsupportedFeatureError
from lmux.mock import MockProvider
from lmux.protocols import AsyncCloseable, CompletionProvider, EmbeddingProvider
from lmux.registry import Registry
from lmux.types import (
    BaseProviderParams,
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    ResponseResponse,
    Tool,
    Usage,
    UserMessage,
)

# MARK: Fixtures


@pytest.fixture
def chat_response() -> ChatResponse:
    return ChatResponse(
        content="Hello!",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03),
        model="gpt-4o",
        provider="mock",
    )


@pytest.fixture
def chat_chunks() -> list[ChatChunk]:
    return [
        ChatChunk(delta="Hel"),
        ChatChunk(delta="lo!"),
        ChatChunk(finish_reason="stop", usage=Usage(input_tokens=10, output_tokens=5)),
    ]


@pytest.fixture
def embed_response() -> EmbeddingResponse:
    return EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        usage=Usage(input_tokens=5, output_tokens=0),
        cost=None,
        model="text-embedding-3-small",
        provider="mock",
    )


@pytest.fixture
def response_response() -> ResponseResponse:
    return ResponseResponse(
        id="resp_123",
        output_text="Hi!",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=None,
        model="gpt-4o",
        provider="mock",
    )


@pytest.fixture
def mock_provider(
    chat_response: ChatResponse,
    chat_chunks: list[ChatChunk],
    embed_response: EmbeddingResponse,
    response_response: ResponseResponse,
) -> MockProvider:
    return MockProvider(
        chat_responses=[chat_response],
        chat_stream_responses=[chat_chunks],
        embed_responses=[embed_response],
        response_responses=[response_response],
    )


@pytest.fixture
def registry(mock_provider: MockProvider) -> Registry:
    reg = Registry()
    reg.register("mock", mock_provider)
    return reg


# MARK: Resolve


class TestResolve:
    def test_valid_prefix_model(self, registry: Registry) -> None:
        result = registry.chat("mock/gpt-4o", [UserMessage(content="Hi")])
        assert result.content == "Hello!"

    def test_missing_slash_raises(self, registry: Registry) -> None:
        with pytest.raises(InvalidRequestError, match="prefix/model"):
            registry.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_unregistered_prefix_raises(self, registry: Registry) -> None:
        with pytest.raises(InvalidRequestError, match="No provider registered"):
            registry.chat("unknown/gpt-4o", [UserMessage(content="Hi")])


# MARK: Chat


class TestChat:
    def test_routes_to_provider(self, registry: Registry) -> None:
        result = registry.chat("mock/gpt-4o", [UserMessage(content="Hi")])
        assert result.content == "Hello!"

    def test_passes_params_through(self, registry: Registry) -> None:
        result = registry.chat(
            "mock/gpt-4o",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
        )
        assert result.content == "Hello!"


class TestAchat:
    async def test_routes_to_provider(self, registry: Registry) -> None:
        result = await registry.achat("mock/gpt-4o", [UserMessage(content="Hi")])
        assert result.content == "Hello!"


class TestChatStream:
    def test_yields_chunks(self, registry: Registry) -> None:
        chunks = list(registry.chat_stream("mock/gpt-4o", [UserMessage(content="Hi")]))
        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo!"


class TestAchatStream:
    async def test_yields_chunks(self, registry: Registry) -> None:
        chunks = [chunk async for chunk in registry.achat_stream("mock/gpt-4o", [UserMessage(content="Hi")])]
        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"


# MARK: Embed


class TestEmbed:
    def test_routes_to_provider(self, registry: Registry) -> None:
        result = registry.embed("mock/text-embedding-3-small", "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]

    def test_passes_dimensions(self, registry: Registry) -> None:
        result = registry.embed("mock/text-embedding-3-small", "hello", dimensions=256)
        assert result.embeddings == [[0.1, 0.2, 0.3]]


class TestAembed:
    async def test_routes_to_provider(self, registry: Registry) -> None:
        result = await registry.aembed("mock/text-embedding-3-small", "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]

    async def test_passes_dimensions(self, registry: Registry) -> None:
        result = await registry.aembed("mock/text-embedding-3-small", "hello", dimensions=256)
        assert result.embeddings == [[0.1, 0.2, 0.3]]


# MARK: Responses


class TestCreateResponse:
    def test_routes_to_provider(self, registry: Registry) -> None:
        result = registry.create_response("mock/gpt-4o", "Hello")
        assert result.output_text == "Hi!"


class TestAcreateResponse:
    async def test_routes_to_provider(self, registry: Registry) -> None:
        result = await registry.acreate_response("mock/gpt-4o", "Hello")
        assert result.output_text == "Hi!"


# MARK: Unsupported Feature


class CompletionOnlyProvider(CompletionProvider[None]):
    """A provider that only supports chat, not embed or responses."""

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> ChatResponse:
        return ChatResponse(  # pragma: no cover
            content="ok", usage=Usage(input_tokens=1, output_tokens=1), cost=None, model=model, provider="test"
        )

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> ChatResponse:
        return ChatResponse(  # pragma: no cover
            content="ok", usage=Usage(input_tokens=1, output_tokens=1), cost=None, model=model, provider="test"
        )

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> Iterator[ChatChunk]:
        yield ChatChunk(delta="ok")  # pragma: no cover

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> AsyncIterator[ChatChunk]:
        yield ChatChunk(delta="ok")  # pragma: no cover


class EmbeddingOnlyProvider(EmbeddingProvider[None]):
    """A provider that only supports embed, not chat or responses."""

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: None = None,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(  # pragma: no cover
            embeddings=[[0.1]], usage=Usage(input_tokens=1, output_tokens=0), cost=None, model=model, provider="test"
        )

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        dimensions: int | None = None,
        provider_params: None = None,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(  # pragma: no cover
            embeddings=[[0.1]], usage=Usage(input_tokens=1, output_tokens=0), cost=None, model=model, provider="test"
        )


class TestUnsupportedFeature:
    def test_embed_on_completion_only_raises(self) -> None:
        reg = Registry()
        reg.register("test", CompletionOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support embeddings"):
            reg.embed("test/model", "hello")

    async def test_aembed_on_completion_only_raises(self) -> None:
        reg = Registry()
        reg.register("test", CompletionOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support embeddings"):
            await reg.aembed("test/model", "hello")

    def test_create_response_on_completion_only_raises(self) -> None:
        reg = Registry()
        reg.register("test", CompletionOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support the Responses API"):
            reg.create_response("test/model", "hello")

    async def test_acreate_response_on_completion_only_raises(self) -> None:
        reg = Registry()
        reg.register("test", CompletionOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support the Responses API"):
            await reg.acreate_response("test/model", "hello")

    def test_chat_on_embedding_only_raises(self) -> None:
        reg = Registry()
        reg.register("emb", EmbeddingOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support chat"):
            reg.chat("emb/model", [UserMessage(content="Hi")])

    async def test_achat_on_embedding_only_raises(self) -> None:
        reg = Registry()
        reg.register("emb", EmbeddingOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support chat"):
            await reg.achat("emb/model", [UserMessage(content="Hi")])

    def test_chat_stream_on_embedding_only_raises(self) -> None:
        reg = Registry()
        reg.register("emb", EmbeddingOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support chat"):
            list(reg.chat_stream("emb/model", [UserMessage(content="Hi")]))

    async def test_achat_stream_on_embedding_only_raises(self) -> None:
        reg = Registry()
        reg.register("emb", EmbeddingOnlyProvider())
        with pytest.raises(UnsupportedFeatureError, match="does not support chat"):
            async for _ in reg.achat_stream("emb/model", [UserMessage(content="Hi")]):
                pass  # pragma: no cover


# MARK: Provider Params Resolution


class FakeParams(BaseProviderParams):
    tag: str = "default"


class RecordingProvider(CompletionProvider[FakeParams]):
    """A provider that records the provider_params and reasoning_effort it receives."""

    last_params: BaseProviderParams | None = None
    last_reasoning_effort: Literal["low", "medium", "high"] | None = None

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: FakeParams | None = None,
    ) -> ChatResponse:
        self.last_params = provider_params
        self.last_reasoning_effort = reasoning_effort
        return ChatResponse(
            content="ok", usage=Usage(input_tokens=1, output_tokens=1), cost=None, model=model, provider="rec"
        )

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: FakeParams | None = None,
    ) -> ChatResponse:
        self.last_params = provider_params  # pragma: no cover
        return ChatResponse(  # pragma: no cover
            content="ok", usage=Usage(input_tokens=1, output_tokens=1), cost=None, model=model, provider="rec"
        )

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: FakeParams | None = None,
    ) -> Iterator[ChatChunk]:
        self.last_params = provider_params  # pragma: no cover
        yield ChatChunk(delta="ok")  # pragma: no cover

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
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: FakeParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.last_params = provider_params  # pragma: no cover
        yield ChatChunk(delta="ok")  # pragma: no cover


class _CloseableProvider(MockProvider, AsyncCloseable):
    async def aclose(self) -> None: ...  # pragma: no cover


class TestAclose:
    async def test_closes_closeable_providers(self) -> None:
        mock = MockProvider()
        reg = Registry()
        reg.register("mock", mock)
        # MockProvider doesn't implement AsyncCloseable, so this should be a no-op
        await reg.aclose()

    async def test_closes_async_closeable_provider(self) -> None:
        prov = _CloseableProvider()
        prov.aclose = AsyncMock()
        reg = Registry()
        reg.register("closeable", prov)
        reg.register("mock", MockProvider())

        await reg.aclose()

        prov.aclose.assert_awaited_once()


class TestProviderParamsResolution:
    def test_no_params_no_default(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov)
        reg.chat("rec/model", [UserMessage(content="Hi")])
        assert prov.last_params is None

    def test_default_params_used_when_no_per_call(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov, default_params=FakeParams(tag="from-default"))
        reg.chat("rec/model", [UserMessage(content="Hi")])
        assert isinstance(prov.last_params, FakeParams)
        assert prov.last_params.tag == "from-default"

    def test_per_call_basemodel_overrides_default(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov, default_params=FakeParams(tag="default"))
        reg.chat("rec/model", [UserMessage(content="Hi")], provider_params=FakeParams(tag="override"))
        assert isinstance(prov.last_params, FakeParams)
        assert prov.last_params.tag == "override"

    def test_dict_params_resolved_by_prefix(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov)
        params = {"rec": FakeParams(tag="from-dict"), "other": FakeParams(tag="wrong")}
        reg.chat("rec/model", [UserMessage(content="Hi")], provider_params=params)
        assert isinstance(prov.last_params, FakeParams)
        assert prov.last_params.tag == "from-dict"

    def test_dict_missing_prefix_falls_back_to_default(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov, default_params=FakeParams(tag="fallback"))
        params: dict[str, BaseProviderParams] = {"other": FakeParams(tag="wrong")}
        reg.chat("rec/model", [UserMessage(content="Hi")], provider_params=params)
        assert isinstance(prov.last_params, FakeParams)
        assert prov.last_params.tag == "fallback"

    def test_dict_entry_overrides_default(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov, default_params=FakeParams(tag="default"))
        params = {"rec": FakeParams(tag="dict-wins")}
        reg.chat("rec/model", [UserMessage(content="Hi")], provider_params=params)
        assert isinstance(prov.last_params, FakeParams)
        assert prov.last_params.tag == "dict-wins"

    def test_dict_missing_prefix_no_default_gives_none(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov)
        params: dict[str, BaseProviderParams] = {"other": FakeParams(tag="wrong")}
        reg.chat("rec/model", [UserMessage(content="Hi")], provider_params=params)
        assert prov.last_params is None

    def test_reasoning_effort_passed_through(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov)
        reg.chat("rec/model", [UserMessage(content="Hi")], reasoning_effort="high")
        assert prov.last_reasoning_effort == "high"

    def test_reregister_without_default_clears_old_default(self) -> None:
        prov = RecordingProvider()
        reg = Registry()
        reg.register("rec", prov, default_params=FakeParams(tag="old"))
        reg.register("rec", prov)
        reg.chat("rec/model", [UserMessage(content="Hi")])
        assert prov.last_params is None
