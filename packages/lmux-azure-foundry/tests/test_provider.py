"""Tests for Azure AI Foundry provider."""

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage as EmbUsage
from openai.types.embedding import Embedding

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    Tool,
    UserMessage,
)
from lmux_azure_foundry import preload
from lmux_azure_foundry.auth import AzureAdToken
from lmux_azure_foundry.params import AzureFoundryParams
from lmux_azure_foundry.provider import AzureFoundryProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider that returns an API key."""

    def get_credentials(self) -> str:
        return "fake-api-key"

    async def aget_credentials(self) -> str:
        return "fake-api-key"


class FakeTokenAuth:
    """Fake auth provider that returns an AzureAdToken."""

    def get_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="fake-ad-token")  # noqa: S106

    async def aget_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="fake-ad-token")  # pragma: no cover  # noqa: S106


class FakeTokenProviderAuth:
    """Fake auth provider that returns a token provider callable."""

    @staticmethod
    def _provider() -> str:
        return "fresh-token"  # pragma: no cover

    def get_credentials(self) -> Callable[[], str]:
        return self._provider

    async def aget_credentials(self) -> Callable[[], str]:
        return self._provider  # pragma: no cover


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="Hello!", role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def stream_chunks() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[ChunkChoice(delta=ChoiceDelta(content="Hel"), index=0, finish_reason=None)],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[ChunkChoice(delta=ChoiceDelta(content="lo!"), index=0, finish_reason=None)],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[ChunkChoice(delta=ChoiceDelta(), index=0, finish_reason="stop")],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]


@pytest.fixture
def embedding_response() -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=EmbUsage(prompt_tokens=5, total_tokens=5),
    )


@pytest.fixture
def mock_sync_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_sync_create(mock_sync_client: MagicMock) -> Iterator[MagicMock]:
    with patch("lmux_azure_foundry.provider.create_sync_client", return_value=mock_sync_client) as mock_create:
        yield mock_create


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_create: MagicMock) -> AzureFoundryProvider:
    return AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=fake_auth)


@pytest.fixture
def mock_async_client() -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.embeddings.create = AsyncMock()
    return mock


@pytest.fixture
def mock_async_create(mock_async_client: MagicMock) -> Iterator[MagicMock]:
    with patch("lmux_azure_foundry.provider.create_async_client", return_value=mock_async_client) as mock_create:
        yield mock_create


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_create: MagicMock) -> AzureFoundryProvider:
    return AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=fake_auth)


@pytest.fixture
def bad_request_error() -> openai.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return openai.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def auth_error() -> openai.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return openai.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def server_error() -> openai.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return openai.InternalServerError(message="test error", response=response, body=None)


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.provider == "azure-foundry"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        mock_sync_client.chat.completions.create.assert_called_once()
        mock_sync_client.embeddings.create.assert_not_called()

    def test_chat_with_params(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

    def test_chat_with_tools(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], tools=tools)

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

    def test_chat_with_response_format(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat())

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            response_format={"type": "json_object"},
        )

    def test_chat_with_provider_params(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(seed=42, user="u1"),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            seed=42,
            user="u1",
        )

    def test_chat_with_reasoning_effort(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "o3",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(reasoning_effort="high"),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="o3",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            reasoning={"effort": "high"},
        )

    def test_chat_exception_mapping(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        bad_request_error: openai.BadRequestError,
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_chat_cost_calculated(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.total_cost > 0

    def test_chat_data_zone_multiplier(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result_global = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result_dz = sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )

        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_chat_regional_multiplier(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result_global = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result_regional = sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="regional"),
        )

        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_chat_no_multiplier_without_params(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result1 = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result2 = sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="global"),
        )

        assert result1.cost is not None
        assert result2.cost is not None
        assert result1.cost.total_cost == pytest.approx(result2.cost.total_cost)

    def test_chat_no_multiplier_with_none_cost(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock
    ) -> None:
        unknown_completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="totally-unknown-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_sync_client.chat.completions.create.return_value = unknown_completion

        result = sync_provider.chat(
            "totally-unknown-model",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )

        assert result.cost is None


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: AzureFoundryProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        result = await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "azure-foundry"
        mock_async_client.chat.completions.create.assert_awaited_once()
        mock_async_client.embeddings.create.assert_not_called()

    async def test_achat_exception_mapping(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        auth_error: openai.AuthenticationError,
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo!"
        assert chunks[2].finish_reason == "stop"

    def test_cost_on_final_chunk(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_stream_data_zone_multiplier(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks_global = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks_dz = list(
            sync_provider.chat_stream(
                "gpt-4o",
                [UserMessage(content="Hi")],
                provider_params=AzureFoundryParams(deployment_type="data_zone"),
            )
        )

        assert chunks_global[2].cost is not None
        assert chunks_dz[2].cost is not None
        assert chunks_dz[2].cost.total_cost == pytest.approx(chunks_global[2].cost.total_cost * 1.1)

    def test_stream_exception_on_create(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        server_error: openai.InternalServerError,
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
        server_error: openai.InternalServerError,
    ) -> None:
        def _failing_iter() -> Any:  # noqa: ANN401
            yield stream_chunks[0]
            raise server_error

        mock_sync_client.chat.completions.create.return_value = _failing_iter()

        with pytest.raises(ProviderError, match="test error"):
            list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        async def _async_iter() -> Any:  # noqa: ANN401
            for chunk in stream_chunks:
                yield chunk

        mock_async_client.chat.completions.create.return_value = _async_iter()

        chunks = [chunk async for chunk in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")])]

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].cost is not None

    async def test_achat_stream_data_zone_multiplier(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        async def _async_iter_global() -> Any:  # noqa: ANN401
            for chunk in stream_chunks:
                yield chunk

        async def _async_iter_dz() -> Any:  # noqa: ANN401
            for chunk in stream_chunks:
                yield chunk

        mock_async_client.chat.completions.create.return_value = _async_iter_global()
        chunks_global = [chunk async for chunk in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")])]

        mock_async_client.chat.completions.create.return_value = _async_iter_dz()
        chunks_dz = [
            chunk
            async for chunk in async_provider.achat_stream(
                "gpt-4o",
                [UserMessage(content="Hi")],
                provider_params=AzureFoundryParams(deployment_type="data_zone"),
            )
        ]

        assert chunks_global[2].cost is not None
        assert chunks_dz[2].cost is not None
        assert chunks_dz[2].cost.total_cost == pytest.approx(chunks_global[2].cost.total_cost * 1.1)

    async def test_exception_on_create(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        server_error: openai.InternalServerError,
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_during_iteration(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
        server_error: openai.InternalServerError,
    ) -> None:
        async def _failing_async_iter() -> Any:  # noqa: ANN401
            yield stream_chunks[0]
            raise server_error

        mock_async_client.chat.completions.create.return_value = _failing_async_iter()

        with pytest.raises(ProviderError, match="test error"):
            async for _ in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass


# MARK: Embed


class TestEmbed:
    def test_basic_embed(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        result = sync_provider.embed("text-embedding-3-small", "hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "azure-foundry"
        mock_sync_client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input="hello")
        mock_sync_client.chat.completions.create.assert_not_called()

    def test_embed_list_input(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        sync_provider.embed("text-embedding-3-small", ["hello", "world"])

        mock_sync_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["hello", "world"]
        )

    def test_embed_exception_mapping(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        bad_request_error: openai.BadRequestError,
    ) -> None:
        mock_sync_client.embeddings.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.embed("text-embedding-3-small", "hello")

    async def test_aembed(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_async_client.embeddings.create.return_value = embedding_response

        result = await async_provider.aembed("text-embedding-3-small", "hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        mock_async_client.embeddings.create.assert_awaited_once_with(model="text-embedding-3-small", input="hello")
        mock_async_client.chat.completions.create.assert_not_called()

    def test_embed_with_provider_params(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        sync_provider.embed("text-embedding-3-small", "hello", provider_params=AzureFoundryParams(user="u1"))

        mock_sync_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="hello", user="u1"
        )

    async def test_aembed_with_provider_params(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_async_client.embeddings.create.return_value = embedding_response

        await async_provider.aembed("text-embedding-3-small", "hello", provider_params=AzureFoundryParams(user="u1"))

        mock_async_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small", input="hello", user="u1"
        )

    async def test_aembed_exception_mapping(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        bad_request_error: openai.BadRequestError,
    ) -> None:
        mock_async_client.embeddings.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            await async_provider.aembed("text-embedding-3-small", "hello")

    def test_embed_data_zone_multiplier(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        result_global = sync_provider.embed("text-embedding-3-small", "hello")
        result_dz = sync_provider.embed(
            "text-embedding-3-small",
            "hello",
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )

        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    async def test_aembed_data_zone_multiplier(
        self,
        async_provider: AzureFoundryProvider,
        mock_async_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_async_client.embeddings.create.return_value = embedding_response

        result_global = await async_provider.aembed("text-embedding-3-small", "hello")
        result_dz = await async_provider.aembed(
            "text-embedding-3-small",
            "hello",
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )

        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_embed_no_multiplier_without_params(
        self,
        sync_provider: AzureFoundryProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        result1 = sync_provider.embed("text-embedding-3-small", "hello")
        result2 = sync_provider.embed(
            "text-embedding-3-small",
            "hello",
            provider_params=AzureFoundryParams(deployment_type="global"),
        )

        assert result1.cost is not None
        assert result2.cost is not None
        assert result1.cost.total_cost == pytest.approx(result2.cost.total_cost)


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        sync_provider.chat("gpt-4o", [UserMessage(content="Hi again")])

        assert mock_sync_client.chat.completions.create.call_count == 2

    async def test_async_client_reused(
        self, async_provider: AzureFoundryProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])
        await async_provider.achat("gpt-4o", [UserMessage(content="Hi again")])

        assert mock_async_client.chat.completions.create.call_count == 2

    def test_endpoint_and_api_version_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(
            endpoint="https://my.openai.azure.com/", auth=fake_auth, api_version="2025-01-01"
        )
        provider.chat("gpt-4o", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            credential="fake-api-key",
            azure_endpoint="https://my.openai.azure.com/",
            api_version="2025-01-01",
            timeout=None,
            max_retries=None,
        )

    def test_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(
            endpoint="https://test.openai.azure.com/", auth=fake_auth, timeout=30.0, max_retries=5
        )
        provider.chat("gpt-4o", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            credential="fake-api-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            timeout=30.0,
            max_retries=5,
        )

    def test_create_sync_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=fake_auth)
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        provider.chat("gpt-4o", [UserMessage(content="Hi again")])

        mock_sync_create.assert_called_once()

    async def test_create_async_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=fake_auth)
        await provider.achat("gpt-4o", [UserMessage(content="Hi")])
        await provider.achat("gpt-4o", [UserMessage(content="Hi again")])

        mock_async_create.assert_called_once()

    async def test_async_endpoint_and_api_version_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(
            endpoint="https://my.openai.azure.com/", auth=fake_auth, api_version="2025-01-01"
        )
        await provider.achat("gpt-4o", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(
            credential="fake-api-key",
            azure_endpoint="https://my.openai.azure.com/",
            api_version="2025-01-01",
            timeout=None,
            max_retries=None,
        )

    async def test_async_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(
            endpoint="https://test.openai.azure.com/", auth=fake_auth, timeout=30.0, max_retries=5
        )
        await provider.achat("gpt-4o", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(
            credential="fake-api-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            timeout=30.0,
            max_retries=5,
        )

    def test_azure_ad_token_credential(
        self,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=FakeTokenAuth())
        provider.chat("gpt-4o", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            credential=AzureAdToken(token="fake-ad-token"),  # noqa: S106
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            timeout=None,
            max_retries=None,
        )

    def test_token_provider_credential(
        self,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        auth = FakeTokenProviderAuth()
        provider = AzureFoundryProvider(endpoint="https://test.openai.azure.com/", auth=auth)
        provider.chat("gpt-4o", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            credential=FakeTokenProviderAuth._provider,  # pyright: ignore[reportPrivateUsage]
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            timeout=None,
            max_retries=None,
        )

    def test_default_auth_used_when_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "env-key")
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = AzureFoundryProvider(endpoint="https://test.openai.azure.com/")
        provider.chat("gpt-4o", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            credential="env-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-12-01-preview",
            timeout=None,
            max_retries=None,
        )


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams())

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert "seed" not in call_kwargs
        assert "user" not in call_kwargs
        assert "deployment_type" not in call_kwargs

    def test_all_params(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        params = AzureFoundryParams(reasoning_effort="low", seed=42, user="u1")

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "low"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["user"] == "u1"
        assert "deployment_type" not in call_kwargs

    def test_deployment_type_not_forwarded(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert "deployment_type" not in call_kwargs


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock
    ) -> None:
        custom_completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="ft:custom-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        mock_sync_client.chat.completions.create.return_value = custom_completion

        sync_provider.register_pricing(
            "ft:custom-model",
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat("ft:custom-model", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        custom_pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
        )
        sync_provider.register_pricing("gpt-4o", custom_pricing)
        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: AzureFoundryProvider, mock_sync_client: MagicMock
    ) -> None:
        unknown_completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="totally-unknown-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_sync_client.chat.completions.create.return_value = unknown_completion

        result = sync_provider.chat("totally-unknown-model", [UserMessage(content="Hi")])

        assert result.cost is None


# MARK: Preload


class TestPreload:
    def test_preload_imports_openai(self) -> None:
        preload()  # should not raise
