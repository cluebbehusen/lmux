"""Tests for Anthropic provider."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    UnsupportedFeatureError,
)
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextResponseFormat,
    Tool,
    UserMessage,
)
from lmux_anthropic import preload
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import AnthropicProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider for testing."""

    def get_credentials(self) -> str:
        return "sk-ant-fake-key"

    async def aget_credentials(self) -> str:
        return "sk-ant-fake-key"


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


def _make_message_response(
    *,
    text: str = "Hello!",
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    message = MagicMock()
    message.content = [text_block]
    message.model = model
    message.stop_reason = stop_reason
    message.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return message


@pytest.fixture
def message_response() -> MagicMock:
    return _make_message_response()


def _make_stream_events() -> list[MagicMock]:
    start_event = MagicMock()
    start_event.type = "message_start"
    start_event.message.usage = MagicMock(
        input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )

    text_delta = MagicMock()
    text_delta.type = "content_block_delta"
    text_delta.delta.type = "text_delta"
    text_delta.delta.text = "Hel"
    text_delta.index = 0

    text_delta2 = MagicMock()
    text_delta2.type = "content_block_delta"
    text_delta2.delta.type = "text_delta"
    text_delta2.delta.text = "lo!"
    text_delta2.index = 0

    delta_event = MagicMock()
    delta_event.type = "message_delta"
    delta_event.delta.stop_reason = "end_turn"
    delta_event.usage.output_tokens = 5

    return [start_event, text_delta, text_delta2, delta_event]


@pytest.fixture
def stream_events() -> list[MagicMock]:
    return _make_stream_events()


@pytest.fixture
def mock_sync_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_sync_create(mock_sync_client: MagicMock) -> Iterator[MagicMock]:
    with patch("lmux_anthropic.provider.create_sync_client", return_value=mock_sync_client) as mock_create:
        yield mock_create


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_create: MagicMock) -> AnthropicProvider:
    return AnthropicProvider(auth=fake_auth)


@pytest.fixture
def mock_async_client() -> MagicMock:
    mock = MagicMock()
    mock.messages.create = AsyncMock()
    return mock


@pytest.fixture
def mock_async_create(mock_async_client: MagicMock) -> Iterator[MagicMock]:
    with patch("lmux_anthropic.provider.create_async_client", return_value=mock_async_client) as mock_create:
        yield mock_create


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_create: MagicMock) -> AnthropicProvider:
    return AnthropicProvider(auth=fake_auth)


@pytest.fixture
def bad_request_error() -> anthropic.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return anthropic.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def auth_error() -> anthropic.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return anthropic.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def server_error() -> anthropic.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return anthropic.InternalServerError(message="test error", response=response, body=None)


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.model == "claude-sonnet-4-6"
        assert result.provider == "anthropic"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        mock_sync_client.messages.create.assert_called_once()

    def test_chat_with_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

        mock_sync_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            stream=False,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["END"],
        )

    def test_chat_default_max_tokens(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_chat_explicit_max_tokens_overrides_default(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], max_tokens=200)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 200

    def test_chat_with_system_message(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [SystemMessage(content="Be helpful."), UserMessage(content="Hi")])

        mock_sync_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            stream=False,
            system="Be helpful.",
        )

    def test_chat_with_tools(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], tools=tools)

        mock_sync_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            stream=False,
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        )

    def test_chat_with_json_schema_response_format(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        rf = JsonSchemaResponseFormat(name="person", json_schema={"type": "object"})
        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], response_format=rf)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}

    def test_chat_text_response_format_is_noop(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], response_format=TextResponseFormat())

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs

    def test_chat_json_object_raises(self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock) -> None:
        with pytest.raises(UnsupportedFeatureError, match="JsonObjectResponseFormat"):
            sync_provider.chat(
                "claude-sonnet-4-6",
                [UserMessage(content="Hi")],
                response_format=JsonObjectResponseFormat(),
            )

    def test_chat_with_provider_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(
                thinking={"type": "enabled", "budget_tokens": 10000},
                top_k=40,
                service_tier="auto",
            ),
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["service_tier"] == "auto"

    def test_chat_with_stop_string(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], stop="STOP")

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["stop_sequences"] == ["STOP"]

    def test_chat_exception_mapping(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        bad_request_error: anthropic.BadRequestError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

    def test_chat_cost_calculated(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.total_cost > 0

    def test_chat_us_inference_multiplier(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result_standard = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        result_us = sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(inference_geo="us"),
        )

        assert result_standard.cost is not None
        assert result_us.cost is not None
        assert result_us.cost.total_cost == pytest.approx(result_standard.cost.total_cost * 1.1)

    def test_chat_no_multiplier_without_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        result_empty = sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(),
        )

        assert result.cost is not None
        assert result_empty.cost is not None
        assert result.cost.total_cost == result_empty.cost.total_cost


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: AnthropicProvider, mock_async_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        result = await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "anthropic"
        mock_async_client.messages.create.assert_awaited_once()

    async def test_achat_exception_mapping(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
        auth_error: anthropic.AuthenticationError,
    ) -> None:
        mock_async_client.messages.create.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        mock_sync_client.messages.create.return_value = iter(stream_events)

        chunks = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo!"
        assert chunks[2].finish_reason == "end_turn"

    def test_cost_on_final_chunk(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        mock_sync_client.messages.create.return_value = iter(stream_events)

        chunks = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_stream_with_content_block_start(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
    ) -> None:
        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.usage = MagicMock(
            input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )

        text_block_start = MagicMock()
        text_block_start.type = "content_block_start"
        text_block_start.content_block.type = "text"

        block_start = MagicMock()
        block_start.type = "content_block_start"
        block_start.content_block.type = "tool_use"
        block_start.content_block.id = "call_1"
        block_start.content_block.name = "get_weather"
        block_start.index = 0

        unknown_delta = MagicMock()
        unknown_delta.type = "content_block_delta"
        unknown_delta.delta.type = "thinking_delta"

        unknown_event = MagicMock()
        unknown_event.type = "content_block_stop"

        delta_event = MagicMock()
        delta_event.type = "message_delta"
        delta_event.delta.stop_reason = "tool_use"
        delta_event.usage.output_tokens = 5

        mock_sync_client.messages.create.return_value = iter(
            [start_event, text_block_start, block_start, unknown_delta, unknown_event, delta_event]
        )

        chunks = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

        assert len(chunks) == 2
        assert chunks[0].tool_call_deltas is not None
        assert chunks[0].tool_call_deltas[0].function is not None
        assert chunks[0].tool_call_deltas[0].function.name == "get_weather"
        assert chunks[1].finish_reason == "tool_use"

    def test_stream_exception_on_create(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        server_error: anthropic.InternalServerError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        stream_events: list[MagicMock],
        server_error: anthropic.InternalServerError,
    ) -> None:
        def _failing_iter() -> Any:  # noqa: ANN401
            yield stream_events[0]
            yield stream_events[1]
            raise server_error

        mock_sync_client.messages.create.return_value = _failing_iter()

        with pytest.raises(ProviderError, match="test error"):
            list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        async def _async_iter() -> Any:  # noqa: ANN401
            for event in stream_events:
                yield event

        mock_async_client.messages.create.return_value = _async_iter()

        chunks = [
            chunk async for chunk in async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[2].finish_reason == "end_turn"
        assert chunks[2].cost is not None

    async def test_stream_with_content_block_start(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
    ) -> None:
        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.usage = MagicMock(
            input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
        )

        text_block_start = MagicMock()
        text_block_start.type = "content_block_start"
        text_block_start.content_block.type = "text"

        block_start = MagicMock()
        block_start.type = "content_block_start"
        block_start.content_block.type = "tool_use"
        block_start.content_block.id = "call_1"
        block_start.content_block.name = "get_weather"
        block_start.index = 0

        unknown_delta = MagicMock()
        unknown_delta.type = "content_block_delta"
        unknown_delta.delta.type = "thinking_delta"

        unknown_event = MagicMock()
        unknown_event.type = "content_block_stop"

        delta_event = MagicMock()
        delta_event.type = "message_delta"
        delta_event.delta.stop_reason = "tool_use"
        delta_event.usage.output_tokens = 5

        async def _async_iter() -> Any:  # noqa: ANN401
            for event in [start_event, text_block_start, block_start, unknown_delta, unknown_event, delta_event]:
                yield event

        mock_async_client.messages.create.return_value = _async_iter()

        chunks = [
            chunk async for chunk in async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 2
        assert chunks[0].tool_call_deltas is not None
        assert chunks[0].tool_call_deltas[0].function is not None
        assert chunks[0].tool_call_deltas[0].function.name == "get_weather"
        assert chunks[1].finish_reason == "tool_use"

    async def test_exception_on_create(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
        server_error: anthropic.InternalServerError,
    ) -> None:
        mock_async_client.messages.create.side_effect = server_error

        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_during_iteration(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
        stream_events: list[MagicMock],
        server_error: anthropic.InternalServerError,
    ) -> None:
        async def _failing_async_iter() -> Any:  # noqa: ANN401
            yield stream_events[0]
            raise server_error

        mock_async_client.messages.create.return_value = _failing_async_iter()

        with pytest.raises(ProviderError, match="test error"):
            async for _ in async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]):
                pass


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        assert mock_sync_client.messages.create.call_count == 2

    async def test_async_client_reused(
        self, async_provider: AnthropicProvider, mock_async_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        assert mock_async_client.messages.create.call_count == 2

    def test_custom_base_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth, base_url="https://custom.api/v1")
        provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(
            api_key="sk-ant-fake-key", base_url="https://custom.api/v1", timeout=None, max_retries=None
        )

    def test_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once_with(api_key="sk-ant-fake-key", base_url=None, timeout=30.0, max_retries=5)

    def test_create_sync_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth)
        provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        mock_sync_create.assert_called_once()

    async def test_create_async_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth)
        await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        mock_async_create.assert_called_once()

    async def test_async_custom_base_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth, base_url="https://custom.api/v1")
        await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(
            api_key="sk-ant-fake-key", base_url="https://custom.api/v1", timeout=None, max_retries=None
        )

    async def test_async_timeout_and_retries_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(api_key="sk-ant-fake-key", base_url=None, timeout=30.0, max_retries=5)


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], provider_params=AnthropicParams())

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert "metadata" not in call_kwargs
        assert "top_k" not in call_kwargs
        assert "service_tier" not in call_kwargs
        assert "inference_geo" not in call_kwargs

    def test_all_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        params = AnthropicParams(
            thinking={"type": "enabled", "budget_tokens": 5000},
            metadata={"user_id": "u1"},
            top_k=40,
            service_tier="auto",
            inference_geo="us",
        )

        sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert call_kwargs["metadata"] == {"user_id": "u1"}
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["service_tier"] == "auto"
        assert call_kwargs["inference_geo"] == "us"


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock
    ) -> None:
        custom_response = _make_message_response(model="claude-custom-v1", input_tokens=1000, output_tokens=500)
        mock_sync_client.messages.create.return_value = custom_response

        sync_provider.register_pricing(
            "claude-custom-v1",
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat("claude-custom-v1", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        custom_pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
        )
        sync_provider.register_pricing("claude-sonnet-4-6", custom_pricing)
        result = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock
    ) -> None:
        unknown_response = _make_message_response(model="totally-unknown-model")
        mock_sync_client.messages.create.return_value = unknown_response

        result = sync_provider.chat("totally-unknown-model", [UserMessage(content="Hi")])

        assert result.cost is None


# MARK: Custom Default Max Tokens


class TestCustomDefaultMaxTokens:
    def test_custom_default(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth, default_max_tokens=8192)
        provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192


# MARK: Preload


class TestPreload:
    def test_preload_imports_anthropic(self) -> None:
        preload()  # should not raise
