"""Tests for Groq provider."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import groq
import pytest
from groq.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from groq.types.chat.chat_completion import Choice
from groq.types.chat.chat_completion_chunk import Choice as ChunkChoice
from groq.types.chat.chat_completion_chunk import ChoiceDelta
from groq.types.completion_usage import CompletionUsage
from pytest_mock import MockerFixture

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    Tool,
    UserMessage,
)
from lmux_groq import preload
from lmux_groq.params import GroqParams
from lmux_groq.provider import GroqProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider for testing."""

    def get_credentials(self) -> str:
        return "gsk-fake-key"

    async def aget_credentials(self) -> str:
        return "gsk-fake-key"


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
        model="llama-3.3-70b-versatile",
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
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[ChunkChoice(delta=ChoiceDelta(content="lo!"), index=0, finish_reason=None)],
            created=1234567890,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[ChunkChoice(delta=ChoiceDelta(), index=0, finish_reason="stop")],
            created=1234567890,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]


@pytest.fixture
def mock_sync_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_sync_create(mock_sync_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_groq.provider.create_sync_client", return_value=mock_sync_client)


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_create: MagicMock) -> GroqProvider:
    assert mock_sync_create is not None
    return GroqProvider(auth=fake_auth)


@pytest.fixture
def mock_async_client() -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_async_create(mock_async_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_groq.provider.create_async_client", return_value=mock_async_client)


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_create: MagicMock) -> GroqProvider:
    assert mock_async_create is not None
    return GroqProvider(auth=fake_auth)


@pytest.fixture
def bad_request_error() -> groq.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return groq.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def auth_error() -> groq.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return groq.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def server_error() -> groq.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return groq.InternalServerError(message="test error", response=response, body=None)


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.model == "llama-3.3-70b-versatile"
        assert result.provider == "groq"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        mock_sync_client.chat.completions.create.assert_called_once()

    def test_chat_with_params(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "llama-3.3-70b-versatile",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

    def test_chat_with_tools(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")], tools=tools)

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

    def test_chat_with_response_format(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "llama-3.3-70b-versatile",
            [UserMessage(content="Hi")],
            response_format=JsonObjectResponseFormat(),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            response_format={"type": "json_object"},
        )

    def test_chat_with_provider_params(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "llama-3.3-70b-versatile",
            [UserMessage(content="Hi")],
            provider_params=GroqParams(service_tier="flex", seed=42, user="u1"),
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            service_tier="flex",
            seed=42,
            user="u1",
        )

    def test_chat_with_reasoning_effort(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")], reasoning_effort="medium")

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["include_reasoning"] is True

    def test_chat_with_provider_params_reasoning_effort(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "llama-3.3-70b-versatile",
            [UserMessage(content="Hi")],
            provider_params=GroqParams(reasoning_effort="high"),
        )

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert call_kwargs["include_reasoning"] is True

    def test_chat_with_provider_params_reasoning_effort_none(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat(
            "qwen/qwen3-32b",
            [UserMessage(content="Hi")],
            provider_params=GroqParams(reasoning_effort="none"),
        )

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "none"
        assert "include_reasoning" not in call_kwargs

    def test_chat_exception_mapping(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, bad_request_error: groq.BadRequestError
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

    def test_chat_cost_calculated(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        result = sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.total_cost > 0


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: GroqProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        result = await async_provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "groq"
        mock_async_client.chat.completions.create.assert_awaited_once()

    async def test_achat_exception_mapping(
        self, async_provider: GroqProvider, mock_async_client: MagicMock, auth_error: groq.AuthenticationError
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            await async_provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: GroqProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks = list(sync_provider.chat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo!"
        assert chunks[2].finish_reason == "stop"

    def test_cost_on_final_chunk(
        self,
        sync_provider: GroqProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = iter(stream_chunks)

        chunks = list(sync_provider.chat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_stream_exception_on_create(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, server_error: groq.InternalServerError
    ) -> None:
        mock_sync_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: GroqProvider,
        mock_sync_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
        server_error: groq.InternalServerError,
    ) -> None:
        def _failing_iter() -> Any:  # noqa: ANN401
            yield stream_chunks[0]
            raise server_error

        mock_sync_client.chat.completions.create.return_value = _failing_iter()

        with pytest.raises(ProviderError, match="test error"):
            list(sync_provider.chat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]))


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(
        self,
        async_provider: GroqProvider,
        mock_async_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
    ) -> None:
        async def _async_iter() -> Any:  # noqa: ANN401
            for chunk in stream_chunks:
                yield chunk

        mock_async_client.chat.completions.create.return_value = _async_iter()

        chunks = [
            chunk async for chunk in async_provider.achat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].cost is not None

    async def test_exception_on_create(
        self, async_provider: GroqProvider, mock_async_client: MagicMock, server_error: groq.InternalServerError
    ) -> None:
        mock_async_client.chat.completions.create.side_effect = server_error

        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_during_iteration(
        self,
        async_provider: GroqProvider,
        mock_async_client: MagicMock,
        stream_chunks: list[ChatCompletionChunk],
        server_error: groq.InternalServerError,
    ) -> None:
        async def _failing_async_iter() -> Any:  # noqa: ANN401
            yield stream_chunks[0]
            raise server_error

        mock_async_client.chat.completions.create.return_value = _failing_async_iter()

        with pytest.raises(ProviderError, match="test error"):
            async for _ in async_provider.achat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hi")]):
                pass


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi again")])

        assert mock_sync_client.chat.completions.create.call_count == 2

    async def test_async_client_reused(
        self, async_provider: GroqProvider, mock_async_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion

        await async_provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        await async_provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi again")])

        assert mock_async_client.chat.completions.create.call_count == 2

    def test_custom_base_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth, base_url="https://custom.api/v1")
        provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            api_key="gsk-fake-key", base_url="https://custom.api/v1", timeout=None, max_retries=None
        )

    def test_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(api_key="gsk-fake-key", base_url=None, timeout=30.0, max_retries=5)

    def test_create_sync_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth)
        provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi again")])

        mock_sync_create.assert_called_once()

    async def test_create_async_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth)
        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi again")])

        mock_async_create.assert_called_once()

    async def test_async_custom_base_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth, base_url="https://custom.api/v1")
        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(
            api_key="gsk-fake-key", base_url="https://custom.api/v1", timeout=None, max_retries=None
        )

    async def test_async_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(api_key="gsk-fake-key", base_url=None, timeout=30.0, max_retries=5)

    def test_sync_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
    ) -> None:
        mock_sync_create.side_effect = Exception("connection refused")
        provider = GroqProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

    async def test_async_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
    ) -> None:
        mock_async_create.side_effect = Exception("connection refused")
        provider = GroqProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

    @pytest.fixture
    def mock_get_running_loop(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("lmux_groq.provider.asyncio.get_running_loop")

    async def test_achat_recreates_client_on_new_event_loop(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
        mock_get_running_loop: MagicMock,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth)

        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]

        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi again")])

        assert mock_async_create.call_count == 2
        assert mock_get_running_loop.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")], provider_params=GroqParams())

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert "service_tier" not in call_kwargs
        assert "seed" not in call_kwargs
        assert "user" not in call_kwargs

    def test_all_params(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion
        params = GroqParams(service_tier="auto", seed=42, user="u1")

        sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["service_tier"] == "auto"
        assert call_kwargs["seed"] == 42
        assert call_kwargs["user"] == "u1"


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(self, sync_provider: GroqProvider, mock_sync_client: MagicMock) -> None:
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
            model="custom-model-v1",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        mock_sync_client.chat.completions.create.return_value = custom_completion

        sync_provider.register_pricing(
            "custom-model-v1",
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat("custom-model-v1", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock, chat_completion: ChatCompletion
    ) -> None:
        mock_sync_client.chat.completions.create.return_value = chat_completion

        custom_pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
        )
        sync_provider.register_pricing("llama-3.3-70b-versatile", custom_pricing)
        result = sync_provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: GroqProvider, mock_sync_client: MagicMock
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


# MARK: Aclose


class TestAclose:
    async def test_aclose_closes_client(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        chat_completion: ChatCompletion,
    ) -> None:
        mock_async_client.chat.completions.create.return_value = chat_completion
        provider = GroqProvider(auth=fake_auth)

        await provider.achat("llama-3.3-70b-versatile", [UserMessage(content="Hi")])
        await provider.aclose()

        mock_async_create.assert_called_once()
        mock_async_client.close.assert_awaited_once()
        assert provider._async_client is None  # pyright: ignore[reportPrivateUsage]

    async def test_aclose_noop_when_no_client(self, fake_auth: FakeAuth) -> None:
        provider = GroqProvider(auth=fake_auth)
        await provider.aclose()


# MARK: Preload


class TestPreload:
    def test_preload_imports_groq(self) -> None:
        preload()  # should not raise
