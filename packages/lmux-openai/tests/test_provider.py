"""Tests for OpenAI provider."""

from collections.abc import Iterator
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

from lmux.cost import ModelPricing
from lmux.exceptions import AuthenticationError, InvalidRequestError, NotFoundError, ProviderError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    Tool,
    UserMessage,
)
from lmux_openai import preload
from lmux_openai.params import OpenAIParams
from lmux_openai.provider import OpenAIProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider for testing."""

    def get_credentials(self) -> str:
        return "sk-fake-key"

    async def aget_credentials(self) -> str:
        return "sk-fake-key"


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
def responses_mock() -> MagicMock:
    mock_response = MagicMock()
    mock_response.id = "resp_123"
    mock_response.output_text = "Hi!"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.input_tokens_details = None
    return mock_response


@pytest.fixture
def mock_sync_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_client: MagicMock) -> Iterator[OpenAIProvider]:
    with patch("lmux_openai.provider.create_sync_client", return_value=mock_sync_client):
        yield OpenAIProvider(auth=fake_auth)


@pytest.fixture
def mock_async_client() -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.embeddings.create = AsyncMock()
    mock.responses.create = AsyncMock()
    return mock


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_client: MagicMock) -> Iterator[OpenAIProvider]:
    with patch("lmux_openai.provider.create_async_client", return_value=mock_async_client):
        yield OpenAIProvider(auth=fake_auth)


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


@pytest.fixture
def not_found_error() -> openai.NotFoundError:
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return openai.NotFoundError(message="test error", response=response, body=None)


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        mock_sync_client.chat.completions.create.assert_called_once()
        mock_sync_client.embeddings.create.assert_not_called()

    def test_chat_with_params(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
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
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
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
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
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
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=OpenAIParams(service_tier="flex", seed=42, user="u1"),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            service_tier="flex",
            seed=42,
            user="u1",
        )

    def test_chat_with_reasoning_effort(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "o3",
            [UserMessage(content="Hi")],
            provider_params=OpenAIParams(reasoning_effort="high"),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="o3",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            reasoning={"effort": "high"},
        )

    def test_chat_exception_mapping(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, bad_request_error: openai.BadRequestError
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_chat_cost_calculated(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.total_cost > 0


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        result = await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "openai"
        mock_async_client.chat.completions.create.assert_awaited_once()
        mock_async_client.embeddings.create.assert_not_called()

    async def test_achat_exception_mapping(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, auth_error: openai.AuthenticationError
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: OpenAIProvider,
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
        sync_provider: OpenAIProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_stream_exception_on_create(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, server_error: openai.InternalServerError
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: OpenAIProvider,
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
        async_provider: OpenAIProvider,
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

    async def test_exception_on_create(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, server_error: openai.InternalServerError
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_during_iteration(
        self,
        async_provider: OpenAIProvider,
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
        sync_provider: OpenAIProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        result = sync_provider.embed("text-embedding-3-small", "hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "openai"
        mock_sync_client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input="hello")
        mock_sync_client.chat.completions.create.assert_not_called()

    def test_embed_list_input(
        self,
        sync_provider: OpenAIProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        sync_provider.embed("text-embedding-3-small", ["hello", "world"])

        mock_sync_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["hello", "world"]
        )

    def test_embed_exception_mapping(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, bad_request_error: openai.BadRequestError
    ) -> None:
        mock_sync_client.embeddings.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.embed("text-embedding-3-small", "hello")

    async def test_aembed(
        self,
        async_provider: OpenAIProvider,
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
        sync_provider: OpenAIProvider,
        mock_sync_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_sync_client.embeddings.create.return_value = embedding_response

        sync_provider.embed("text-embedding-3-small", "hello", provider_params=OpenAIParams(user="u1"))

        mock_sync_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="hello", user="u1"
        )

    async def test_aembed_with_provider_params(
        self,
        async_provider: OpenAIProvider,
        mock_async_client: MagicMock,
        embedding_response: CreateEmbeddingResponse,
    ) -> None:
        mock_async_client.embeddings.create.return_value = embedding_response

        await async_provider.aembed("text-embedding-3-small", "hello", provider_params=OpenAIParams(user="u1"))

        mock_async_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small", input="hello", user="u1"
        )

    async def test_aembed_exception_mapping(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, bad_request_error: openai.BadRequestError
    ) -> None:
        mock_async_client.embeddings.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            await async_provider.aembed("text-embedding-3-small", "hello")


# MARK: CreateResponse


class TestCreateResponse:
    def test_basic(self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, responses_mock: MagicMock) -> None:
        mock_sync_client.responses.create.return_value = responses_mock

        result = sync_provider.create_response("gpt-4o", "Hello")

        assert result.id == "resp_123"
        assert result.output_text == "Hi!"
        assert result.provider == "openai"
        mock_sync_client.responses.create.assert_called_once_with(model="gpt-4o", input="Hello", stream=False)
        mock_sync_client.chat.completions.create.assert_not_called()

    def test_with_provider_params(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, responses_mock: MagicMock
    ) -> None:
        mock_sync_client.responses.create.return_value = responses_mock

        sync_provider.create_response("gpt-4o", "Hello", provider_params=OpenAIParams(service_tier="flex"))

        mock_sync_client.responses.create.assert_called_once_with(
            model="gpt-4o", input="Hello", stream=False, service_tier="flex"
        )

    def test_exception_mapping(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, not_found_error: openai.NotFoundError
    ) -> None:
        mock_sync_client.responses.create.side_effect = not_found_error

        with pytest.raises(NotFoundError):
            sync_provider.create_response("gpt-4o", "Hello")

    async def test_acreate_response(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, responses_mock: MagicMock
    ) -> None:
        mock_async_client.responses.create.return_value = responses_mock

        result = await async_provider.acreate_response("gpt-4o", "Hello")

        assert result.output_text == "Hi!"
        mock_async_client.responses.create.assert_awaited_once_with(model="gpt-4o", input="Hello", stream=False)

    async def test_acreate_response_with_provider_params(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, responses_mock: MagicMock
    ) -> None:
        mock_async_client.responses.create.return_value = responses_mock

        await async_provider.acreate_response("gpt-4o", "Hello", provider_params=OpenAIParams(service_tier="flex"))

        mock_async_client.responses.create.assert_awaited_once_with(
            model="gpt-4o", input="Hello", stream=False, service_tier="flex"
        )

    async def test_acreate_response_exception_mapping(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, not_found_error: openai.NotFoundError
    ) -> None:
        mock_async_client.responses.create.side_effect = not_found_error

        with pytest.raises(NotFoundError):
            await async_provider.acreate_response("gpt-4o", "Hello")


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        sync_provider.chat("gpt-4o", [UserMessage(content="Hi again")])

        assert mock_sync_client.chat.completions.create.call_count == 2

    async def test_async_client_reused(
        self, async_provider: OpenAIProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])
        await async_provider.achat("gpt-4o", [UserMessage(content="Hi again")])

        assert mock_async_client.chat.completions.create.call_count == 2

    def test_custom_base_url_passed(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth, base_url="https://custom.api/v1")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = chat_completion

        with patch("lmux_openai.provider.create_sync_client", return_value=mock_client) as mock_create:
            provider.chat("gpt-4o", [UserMessage(content="Hi")])

            mock_create.assert_called_once_with(api_key="sk-fake-key", base_url="https://custom.api/v1")

    def test_timeout_and_retries_passed(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = chat_completion

        with patch("lmux_openai.provider.create_sync_client", return_value=mock_client) as mock_create:
            provider.chat("gpt-4o", [UserMessage(content="Hi")])

            mock_create.assert_called_once_with(api_key="sk-fake-key", timeout=30.0, max_retries=5)

    def test_create_sync_client_called_once(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = chat_completion

        with patch("lmux_openai.provider.create_sync_client", return_value=mock_client) as mock_create:
            provider.chat("gpt-4o", [UserMessage(content="Hi")])
            provider.chat("gpt-4o", [UserMessage(content="Hi again")])

            mock_create.assert_called_once()

    async def test_create_async_client_called_once(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=chat_completion)

        with patch("lmux_openai.provider.create_async_client", return_value=mock_client) as mock_create:
            await provider.achat("gpt-4o", [UserMessage(content="Hi")])
            await provider.achat("gpt-4o", [UserMessage(content="Hi again")])

            mock_create.assert_called_once()

    async def test_async_custom_base_url_passed(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth, base_url="https://custom.api/v1")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=chat_completion)

        with patch("lmux_openai.provider.create_async_client", return_value=mock_client) as mock_create:
            await provider.achat("gpt-4o", [UserMessage(content="Hi")])

            mock_create.assert_called_once_with(api_key="sk-fake-key", base_url="https://custom.api/v1")

    async def test_async_timeout_and_retries_passed(self, fake_auth: FakeAuth, chat_completion: ChatCompletion) -> None:
        provider = OpenAIProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=chat_completion)

        with patch("lmux_openai.provider.create_async_client", return_value=mock_client) as mock_create:
            await provider.achat("gpt-4o", [UserMessage(content="Hi")])

            mock_create.assert_called_once_with(api_key="sk-fake-key", timeout=30.0, max_retries=5)


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], provider_params=OpenAIParams())

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert "service_tier" not in call_kwargs
        assert "reasoning" not in call_kwargs
        assert "seed" not in call_kwargs
        assert "user" not in call_kwargs

    def test_all_params(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        params = OpenAIParams(service_tier="auto", reasoning_effort="low", seed=42, user="u1")

        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["service_tier"] == "auto"
        assert call_kwargs["reasoning"] == {"effort": "low"}
        assert call_kwargs["seed"] == 42
        assert call_kwargs["user"] == "u1"


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        """Custom pricing populates cost for models not in built-in PRICING."""
        # Use a model name not in built-in pricing
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
            model="ft:gpt-4o:my-org:custom:id",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        mock_sync_client.chat.completions.create.return_value = custom_completion

        sync_provider.register_pricing(
            "ft:gpt-4o:my-org:custom:id",
            ModelPricing(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000),
        )
        result = sync_provider.chat("ft:gpt-4o:my-org:custom:id", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        """User-registered pricing takes precedence over built-in pricing."""
        mock_sync_client.chat.completions.create.return_value = chat_completion

        # Register custom pricing for a model that already has built-in pricing
        custom_pricing = ModelPricing(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)
        sync_provider.register_pricing("gpt-4o", custom_pricing)
        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: OpenAIProvider, mock_sync_client: MagicMock
    ) -> None:
        """Without registration, unknown models return cost=None."""
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
