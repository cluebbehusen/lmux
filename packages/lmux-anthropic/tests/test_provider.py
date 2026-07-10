"""Tests for Anthropic provider."""

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from pytest_mock import MockerFixture

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
from lmux_anthropic.auth import AnthropicFoundryEnvAuthProvider, AnthropicVertexADCAuthProvider
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import AnthropicFoundryProvider, AnthropicProvider, AnthropicVertexProvider

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
        cache_creation=None,
    )
    return message


@pytest.fixture
def message_response() -> MagicMock:
    return _make_message_response()


def _make_stream_events() -> list[MagicMock]:
    start_event = MagicMock()
    start_event.type = "message_start"
    start_event.message.model = "claude-sonnet-4-6"
    start_event.message.usage = MagicMock(
        input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0, cache_creation=None
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
def mock_sync_create(mock_sync_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_sync_client", return_value=mock_sync_client)


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_create: MagicMock) -> AnthropicProvider:
    assert mock_sync_create  # fixture activates the patch
    return AnthropicProvider(auth=fake_auth)


@pytest.fixture
def mock_async_client() -> MagicMock:
    mock = MagicMock()
    mock.messages.create = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_async_create(mock_async_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_async_client", return_value=mock_async_client)


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_create: MagicMock) -> AnthropicProvider:
    assert mock_async_create  # fixture activates the patch
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


class TestPricingAsOf:
    def test_live_cost_uses_current_date_in_intro_window(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """During the introductory window, live Sonnet 5 cost bills the introductory rate."""
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 7, 1))
        mock_sync_client.messages.create.return_value = _make_message_response(
            model="claude-sonnet-5", input_tokens=1000, output_tokens=500
        )
        response = sync_provider.chat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert response.cost is not None
        assert response.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)
        assert response.cost.output_cost == pytest.approx(500 * 10.0 / 1_000_000)

    def test_live_cost_uses_current_date_after_switch(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After 2026-09-01, live Sonnet 5 cost bills the standard rate."""
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 9, 15))
        mock_sync_client.messages.create.return_value = _make_message_response(
            model="claude-sonnet-5", input_tokens=1000, output_tokens=500
        )
        response = sync_provider.chat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert response.cost is not None
        assert response.cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert response.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_pricing_as_of_override_wins_over_clock(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit pricing_as_of overrides the current date (e.g. for cost replay)."""
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 9, 15))
        mock_sync_client.messages.create.return_value = _make_message_response(
            model="claude-sonnet-5", input_tokens=1000, output_tokens=500
        )
        response = sync_provider.chat(
            "claude-sonnet-5",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(pricing_as_of=date(2026, 7, 1)),
        )
        assert response.cost is not None
        # The override date lands in the intro window, so the introductory rate wins.
        assert response.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)
        assert response.cost.output_cost == pytest.approx(500 * 10.0 / 1_000_000)

    async def test_achat_applies_dated_pricing(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The async path resolves and applies the dated pricing date like the sync path."""
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 7, 1))
        mock_async_client.messages.create.return_value = _make_message_response(
            model="claude-sonnet-5", input_tokens=1000, output_tokens=500
        )
        response = await async_provider.achat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert response.cost is not None
        assert response.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)


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

        _ = sync_provider.chat(
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

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_chat_explicit_max_tokens_overrides_default(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], max_tokens=200)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 200

    def test_chat_with_system_message(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [SystemMessage(content="Be helpful."), UserMessage(content="Hi")])

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
        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], tools=tools)

        mock_sync_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            stream=False,
            tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        )

    def test_chat_with_tool_choice(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], tool_choice="required")

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "any"}

    def test_chat_with_json_schema_response_format(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        rf = JsonSchemaResponseFormat(name="person", json_schema={"type": "object"})
        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], response_format=rf)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {
            "format": {"type": "json_schema", "schema": {"type": "object", "additionalProperties": False}}
        }

    def test_chat_text_response_format_is_noop(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], response_format=TextResponseFormat())

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs

    def test_chat_json_object_raises(self, sync_provider: AnthropicProvider) -> None:
        with pytest.raises(UnsupportedFeatureError, match="JsonObjectResponseFormat is not supported"):
            _ = sync_provider.chat(
                "claude-sonnet-4-6", [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat()
            )

    def test_chat_with_provider_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat(
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

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], stop="STOP")

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["stop_sequences"] == ["STOP"]

    def test_chat_with_reasoning_effort(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        # default max_tokens=4096, so medium (8192) gets capped to 4095
        _ = sync_provider.chat("claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="medium")

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4095}

    def test_chat_with_reasoning_effort_and_high_max_tokens(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat(
            "claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="high", max_tokens=50000
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 32768}
        assert call_kwargs["max_tokens"] == 50000

    def test_chat_reasoning_effort_adaptive_model(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        # A 4.6+ model uses adaptive thinking + output_config.effort, not budget_tokens.
        _ = sync_provider.chat("claude-opus-4-8", [UserMessage(content="Hi")], reasoning_effort="high")

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["output_config"] == {"effort": "high"}

    def test_chat_reasoning_effort_adaptive_merges_response_format(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        # effort must merge into the output_config produced by response_format, not clobber it.
        rf = JsonSchemaResponseFormat(name="person", json_schema={"type": "object"})
        _ = sync_provider.chat(
            "claude-opus-4-8", [UserMessage(content="Hi")], reasoning_effort="medium", response_format=rf
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["output_config"] == {
            "format": {"type": "json_schema", "schema": {"type": "object", "additionalProperties": False}},
            "effort": "medium",
        }

    def test_chat_reasoning_effort_ignored_when_provider_thinking(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        # provider_params.thinking wins: reasoning_effort is fully ignored, leaving no stray effort.
        _ = sync_provider.chat(
            "claude-opus-4-8",
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=AnthropicParams(thinking={"type": "enabled", "budget_tokens": 5000}),
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert "output_config" not in call_kwargs

    def test_chat_exception_mapping(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        bad_request_error: anthropic.BadRequestError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError):
            _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

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
            _ = await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])


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
        assert chunks[2].finish_reason == "stop"
        # The terminal chunk stamps the endpoint identity, mirroring the non-streaming path.
        assert chunks[2].model == "claude-sonnet-4-6"
        assert chunks[2].provider == "anthropic"
        # Interior chunks carry no endpoint identity.
        assert chunks[0].model is None
        assert chunks[0].provider is None

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
        start_event.message.model = "claude-sonnet-4-6"
        start_event.message.usage = MagicMock(
            input_tokens=10,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cache_creation=None,
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

        thinking_delta_event = MagicMock()
        thinking_delta_event.type = "content_block_delta"
        thinking_delta_event.delta.type = "thinking_delta"
        thinking_delta_event.delta.thinking = "Let me think..."

        unknown_delta_event = MagicMock()
        unknown_delta_event.type = "content_block_delta"
        unknown_delta_event.delta.type = "some_future_delta"

        unknown_event = MagicMock()
        unknown_event.type = "content_block_stop"

        delta_event = MagicMock()
        delta_event.type = "message_delta"
        delta_event.delta.stop_reason = "tool_use"
        delta_event.usage.output_tokens = 5

        mock_sync_client.messages.create.return_value = iter(
            [
                start_event,
                text_block_start,
                block_start,
                thinking_delta_event,
                unknown_delta_event,
                unknown_event,
                delta_event,
            ]
        )

        chunks = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

        assert len(chunks) == 3
        assert chunks[0].tool_call_deltas is not None
        assert chunks[0].tool_call_deltas[0].function is not None
        assert chunks[0].tool_call_deltas[0].function.name == "get_weather"
        assert chunks[1].reasoning_delta == "Let me think..."
        assert chunks[2].finish_reason == "tool_calls"

    def test_stream_with_reasoning_effort(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        mock_sync_client.messages.create.return_value = iter(stream_events)

        _ = list(sync_provider.chat_stream("claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="medium"))

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4095}

    def test_stream_exception_on_create(
        self,
        sync_provider: AnthropicProvider,
        mock_sync_client: MagicMock,
        server_error: anthropic.InternalServerError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = server_error

        with pytest.raises(ProviderError):
            _ = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

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
            _ = list(sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))


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
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].cost is not None
        # The terminal chunk stamps the endpoint identity, mirroring the non-streaming path.
        assert chunks[2].model == "claude-sonnet-4-6"
        assert chunks[2].provider == "anthropic"

    async def test_stream_with_content_block_start(
        self,
        async_provider: AnthropicProvider,
        mock_async_client: MagicMock,
    ) -> None:
        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.model = "claude-sonnet-4-6"
        start_event.message.usage = MagicMock(
            input_tokens=10,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cache_creation=None,
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

        thinking_delta_event = MagicMock()
        thinking_delta_event.type = "content_block_delta"
        thinking_delta_event.delta.type = "thinking_delta"
        thinking_delta_event.delta.thinking = "Let me think..."

        unknown_delta_event = MagicMock()
        unknown_delta_event.type = "content_block_delta"
        unknown_delta_event.delta.type = "some_future_delta"

        unknown_event = MagicMock()
        unknown_event.type = "content_block_stop"

        delta_event = MagicMock()
        delta_event.type = "message_delta"
        delta_event.delta.stop_reason = "tool_use"
        delta_event.usage.output_tokens = 5

        async def _async_iter() -> Any:  # noqa: ANN401
            for event in [
                start_event,
                text_block_start,
                block_start,
                thinking_delta_event,
                unknown_delta_event,
                unknown_event,
                delta_event,
            ]:
                yield event

        mock_async_client.messages.create.return_value = _async_iter()

        chunks = [
            chunk async for chunk in async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 3
        assert chunks[0].tool_call_deltas is not None
        assert chunks[0].tool_call_deltas[0].function is not None
        assert chunks[0].tool_call_deltas[0].function.name == "get_weather"
        assert chunks[1].reasoning_delta == "Let me think..."
        assert chunks[2].finish_reason == "tool_calls"

    async def test_stream_with_reasoning_effort(
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
            chunk
            async for chunk in async_provider.achat_stream(
                "claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="medium"
            )
        ]

        assert len(chunks) == 3
        call_kwargs = mock_async_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4095}

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
                pass  # pragma: no cover


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        assert mock_sync_client.messages.create.call_count == 2

    async def test_async_client_reused(
        self, async_provider: AnthropicProvider, mock_async_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        _ = await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        _ = await async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

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
        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

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
        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

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
        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

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
        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

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
        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

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
        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(api_key="sk-ant-fake-key", base_url=None, timeout=30.0, max_retries=5)

    def test_sync_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
    ) -> None:
        mock_sync_create.side_effect = Exception("connection refused")
        provider = AnthropicProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

    async def test_async_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
    ) -> None:
        mock_async_create.side_effect = Exception("connection refused")
        provider = AnthropicProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

    @pytest.fixture
    def mock_get_running_loop(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("lmux_anthropic.provider.asyncio.get_running_loop")

    async def test_achat_recreates_client_on_new_event_loop(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
        mock_get_running_loop: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth)

        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]

        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi again")])

        assert mock_async_create.call_count == 2
        assert mock_get_running_loop.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: AnthropicProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], provider_params=AnthropicParams())

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert "metadata" not in call_kwargs
        assert "top_k" not in call_kwargs
        assert "service_tier" not in call_kwargs
        assert "inference_geo" not in call_kwargs
        assert "cache_control" not in call_kwargs

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
            cache_control={"type": "ephemeral"},
        )

        _ = sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert call_kwargs["metadata"] == {"user_id": "u1"}
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["service_tier"] == "auto"
        assert call_kwargs["inference_geo"] == "us"
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}


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
        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once()
        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192


# MARK: Aclose


class TestAclose:
    async def test_aclose_closes_client(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response
        provider = AnthropicProvider(auth=fake_auth)

        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        await provider.aclose()

        mock_async_create.assert_called_once()
        mock_async_client.close.assert_awaited_once()
        assert provider._async_client is None  # pyright: ignore[reportPrivateUsage]

    async def test_aclose_noop_when_no_client(self, fake_auth: FakeAuth) -> None:
        provider = AnthropicProvider(auth=fake_auth)
        await provider.aclose()


# MARK: Preload


class TestPreload:
    def test_preload_imports_anthropic(self) -> None:
        preload()  # should not raise


# MARK: Vertex


class FakeVertexAuth:
    """Fake Vertex auth provider returning (credentials, project_id), like the ADC provider."""

    def __init__(self) -> None:
        self.credentials: MagicMock = MagicMock()

    def get_credentials(self) -> tuple[MagicMock, str]:
        return (self.credentials, "auth-project")

    async def aget_credentials(self) -> tuple[MagicMock, str]:
        return (self.credentials, "auth-project")


class FakeBareVertexAuth:
    """Fake Vertex auth provider returning bare credentials without a project ID."""

    def __init__(self) -> None:
        self.credentials: MagicMock = MagicMock()

    def get_credentials(self) -> MagicMock:
        return self.credentials

    async def aget_credentials(self) -> MagicMock:
        return self.credentials


@pytest.fixture
def fake_vertex_auth() -> FakeVertexAuth:
    return FakeVertexAuth()


@pytest.fixture
def mock_sync_vertex_create(mock_sync_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_sync_vertex_client", return_value=mock_sync_client)


@pytest.fixture
def mock_async_vertex_create(mock_async_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_async_vertex_client", return_value=mock_async_client)


@pytest.fixture
def vertex_sync_provider(
    fake_vertex_auth: FakeVertexAuth, mock_sync_vertex_create: MagicMock
) -> AnthropicVertexProvider:
    assert mock_sync_vertex_create  # fixture activates the patch
    return AnthropicVertexProvider(auth=fake_vertex_auth, project_id="my-proj", region="us-east5")


@pytest.fixture
def vertex_async_provider(
    fake_vertex_auth: FakeVertexAuth, mock_async_vertex_create: MagicMock
) -> AnthropicVertexProvider:
    assert mock_async_vertex_create  # fixture activates the patch
    return AnthropicVertexProvider(auth=fake_vertex_auth, project_id="my-proj", region="us-east5")


class TestVertexChat:
    def test_basic_chat(
        self,
        vertex_sync_provider: AnthropicVertexProvider,
        mock_sync_client: MagicMock,
        mock_sync_create: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result = vertex_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "anthropic-vertex"
        assert result.cost is not None
        mock_sync_client.messages.create.assert_called_once()
        mock_sync_create.assert_not_called()  # the Anthropic API client factory must never be used

    def test_vertex_model_id_prefix_matches_pricing(
        self, vertex_sync_provider: AnthropicVertexProvider, mock_sync_client: MagicMock
    ) -> None:
        """Vertex @-suffixed model IDs resolve cost via longest-prefix matching."""
        mock_sync_client.messages.create.return_value = _make_message_response(model="claude-sonnet-4-5@20250929")

        result = vertex_sync_provider.chat("claude-sonnet-4-5@20250929", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.total_cost > 0

    def test_api_only_params_dropped(
        self, vertex_sync_provider: AnthropicVertexProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        """service_tier and inference_geo are Anthropic-API-only and must not reach Vertex."""
        mock_sync_client.messages.create.return_value = message_response

        _ = vertex_sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(
                thinking={"type": "enabled", "budget_tokens": 1024},
                metadata={"user_id": "u1"},
                top_k=40,
                service_tier="auto",
                inference_geo="us",
                cache_control={"type": "ephemeral"},
            ),
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert call_kwargs["metadata"] == {"user_id": "u1"}
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}
        assert "service_tier" not in call_kwargs
        assert "inference_geo" not in call_kwargs

    def test_inference_geo_multiplier_not_applied(
        self, vertex_sync_provider: AnthropicVertexProvider, mock_sync_client: MagicMock, message_response: MagicMock
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result_standard = vertex_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        result_us = vertex_sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(inference_geo="us"),
        )

        assert result_standard.cost is not None
        assert result_us.cost is not None
        assert result_us.cost.total_cost == result_standard.cost.total_cost

    def test_regional_premium_applied(
        self,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Regional endpoints bill Claude 4.5+ models at a 10% premium over the global endpoint."""
        assert mock_sync_vertex_create  # fixture activates the patch
        mock_sync_client.messages.create.return_value = message_response
        global_provider = AnthropicVertexProvider(auth=fake_vertex_auth, project_id="p", region="global")
        regional_provider = AnthropicVertexProvider(auth=fake_vertex_auth, project_id="p", region="us-east5")

        result_global = global_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        result_regional = regional_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_no_premium_for_uniform_pricing_models(
        self,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
    ) -> None:
        """Older Claude models are priced uniformly across all Vertex endpoints."""
        assert mock_sync_vertex_create  # fixture activates the patch
        mock_sync_client.messages.create.return_value = _make_message_response(model="claude-3-5-haiku")
        global_provider = AnthropicVertexProvider(auth=fake_vertex_auth, project_id="p", region="global")
        regional_provider = AnthropicVertexProvider(auth=fake_vertex_auth, project_id="p", region="us-east5")

        result_global = global_provider.chat("claude-3-5-haiku", [UserMessage(content="Hi")])
        result_regional = regional_provider.chat("claude-3-5-haiku", [UserMessage(content="Hi")])

        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == result_global.cost.total_cost

    def test_region_falls_back_to_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Without an explicit region, the premium decision uses CLOUD_ML_REGION."""
        assert mock_sync_vertex_create  # fixture activates the patch
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicVertexProvider(auth=fake_vertex_auth, project_id="p")

        monkeypatch.setenv("CLOUD_ML_REGION", "global")
        result_global = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
        result_regional = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_chat_exception_reports_vertex_provider(
        self,
        vertex_sync_provider: AnthropicVertexProvider,
        mock_sync_client: MagicMock,
        bad_request_error: anthropic.BadRequestError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError) as exc_info:
            _ = vertex_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert exc_info.value.provider == "anthropic-vertex"


class TestVertexAchat:
    async def test_basic_achat(
        self,
        vertex_async_provider: AnthropicVertexProvider,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        result = await vertex_async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "anthropic-vertex"
        mock_async_client.messages.create.assert_awaited_once()


class TestVertexChatStream:
    def test_inference_geo_multiplier_not_applied_on_final_chunk(
        self, vertex_sync_provider: AnthropicVertexProvider, mock_sync_client: MagicMock
    ) -> None:
        mock_sync_client.messages.create.side_effect = [iter(_make_stream_events()), iter(_make_stream_events())]

        chunks_standard = list(vertex_sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))
        chunks_us = list(
            vertex_sync_provider.chat_stream(
                "claude-sonnet-4-6",
                [UserMessage(content="Hi")],
                provider_params=AnthropicParams(inference_geo="us"),
            )
        )

        assert chunks_standard[-1].cost is not None
        assert chunks_us[-1].cost is not None
        assert chunks_us[-1].cost.total_cost == chunks_standard[-1].cost.total_cost

    async def test_async_stream_yields_chunks(
        self,
        vertex_async_provider: AnthropicVertexProvider,
        mock_async_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        async def _async_iter() -> Any:  # noqa: ANN401
            for event in stream_events:
                yield event

        mock_async_client.messages.create.return_value = _async_iter()

        chunks = [
            chunk
            async for chunk in vertex_async_provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 3
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].cost is not None
        # The terminal chunk reports the Vertex endpoint identity, not the base provider name.
        assert chunks[2].model == "claude-sonnet-4-6"
        assert chunks[2].provider == "anthropic-vertex"


class TestVertexClientManagement:
    def test_factory_receives_constructor_args(
        self,
        fake_vertex_auth: FakeVertexAuth,
        vertex_sync_provider: AnthropicVertexProvider,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = vertex_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_vertex_create.assert_called_once_with(
            credentials=fake_vertex_auth.credentials,
            project_id="my-proj",
            region="us-east5",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_factory_receives_overrides(
        self,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicVertexProvider(
            auth=fake_vertex_auth,
            project_id="my-proj",
            region="global",
            base_url="https://example.test",
            timeout=30.0,
            max_retries=5,
        )

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_vertex_create.assert_called_once_with(
            credentials=fake_vertex_auth.credentials,
            project_id="my-proj",
            region="global",
            base_url="https://example.test",
            timeout=30.0,
            max_retries=5,
        )

    async def test_async_factory_receives_constructor_args(
        self,
        fake_vertex_auth: FakeVertexAuth,
        vertex_async_provider: AnthropicVertexProvider,
        mock_async_vertex_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        _ = await vertex_async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_vertex_create.assert_called_once_with(
            credentials=fake_vertex_auth.credentials,
            project_id="my-proj",
            region="us-east5",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_project_id_falls_back_to_auth_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Without an explicit project_id or env var, the auth-derived project (e.g. from ADC) is used."""
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicVertexProvider(auth=fake_vertex_auth, region="global")

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_vertex_create.assert_called_once_with(
            credentials=fake_vertex_auth.credentials,
            project_id="auth-project",
            region="global",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_env_project_id_beats_auth_derived(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_vertex_auth: FakeVertexAuth,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """ANTHROPIC_VERTEX_PROJECT_ID wins over the auth-derived project, matching the SDK's precedence."""
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project")
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicVertexProvider(auth=fake_vertex_auth, region="global")

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_vertex_create.assert_called_once_with(
            credentials=fake_vertex_auth.credentials,
            project_id="env-project",
            region="global",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_bare_credentials_auth_supported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_sync_vertex_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Auth providers returning bare credentials (no project tuple) still work."""
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        mock_sync_client.messages.create.return_value = message_response
        bare_auth = FakeBareVertexAuth()
        provider = AnthropicVertexProvider(auth=bare_auth, region="global")

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_vertex_create.assert_called_once_with(
            credentials=bare_auth.credentials,
            project_id=None,
            region="global",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    async def test_bare_credentials_auth_supported_async(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_async_vertex_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Async auth providers returning bare credentials (no project tuple) still work."""
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        mock_async_client.messages.create.return_value = message_response
        bare_auth = FakeBareVertexAuth()
        provider = AnthropicVertexProvider(auth=bare_auth, region="global")

        _ = await provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_vertex_create.assert_called_once_with(
            credentials=bare_auth.credentials,
            project_id=None,
            region="global",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_default_auth_is_adc(self) -> None:
        provider = AnthropicVertexProvider()
        assert isinstance(provider._vertex_auth, AnthropicVertexADCAuthProvider)  # pyright: ignore[reportPrivateUsage]


# MARK: Foundry


class FakeFoundryAuth:
    """Fake Foundry auth provider returning an API key."""

    def get_credentials(self) -> str:
        return "foundry-key"

    async def aget_credentials(self) -> str:
        return "foundry-key"


class FakeFoundryTokenAuth:
    """Fake Foundry auth provider returning an Entra ID token-provider callable."""

    def __init__(self) -> None:
        def _token_provider() -> str:
            return "entra-token"  # pragma: no cover

        self.token_provider: Callable[[], str] = _token_provider

    def get_credentials(self) -> Callable[[], str]:
        return self.token_provider

    async def aget_credentials(self) -> Callable[[], str]:
        return self.token_provider


@pytest.fixture
def fake_foundry_auth() -> FakeFoundryAuth:
    return FakeFoundryAuth()


@pytest.fixture
def mock_sync_foundry_create(mock_sync_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_sync_foundry_client", return_value=mock_sync_client)


@pytest.fixture
def mock_async_foundry_create(mock_async_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_async_foundry_client", return_value=mock_async_client)


@pytest.fixture
def foundry_sync_provider(
    fake_foundry_auth: FakeFoundryAuth, mock_sync_foundry_create: MagicMock
) -> AnthropicFoundryProvider:
    assert mock_sync_foundry_create  # fixture activates the patch
    return AnthropicFoundryProvider(auth=fake_foundry_auth, resource="my-resource")


@pytest.fixture
def foundry_async_provider(
    fake_foundry_auth: FakeFoundryAuth, mock_async_foundry_create: MagicMock
) -> AnthropicFoundryProvider:
    assert mock_async_foundry_create  # fixture activates the patch
    return AnthropicFoundryProvider(auth=fake_foundry_auth, resource="my-resource")


class TestFoundryChat:
    def test_basic_chat(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_client: MagicMock,
        mock_sync_create: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        result = foundry_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "anthropic-foundry"
        assert result.cost is not None
        mock_sync_client.messages.create.assert_called_once()
        mock_sync_create.assert_not_called()  # the Anthropic API client factory must never be used

    def test_api_only_params_dropped(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """service_tier and inference_geo are Anthropic-API-only and must not reach Foundry."""
        mock_sync_client.messages.create.return_value = message_response

        _ = foundry_sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(
                thinking={"type": "enabled", "budget_tokens": 1024},
                top_k=40,
                service_tier="auto",
                inference_geo="us",
            ),
        )

        call_kwargs = mock_sync_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert call_kwargs["top_k"] == 40
        assert "service_tier" not in call_kwargs
        assert "inference_geo" not in call_kwargs

    def test_inference_geo_multiplier_not_applied(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        """Foundry bills Anthropic list prices; no multiplier ever applies."""
        mock_sync_client.messages.create.return_value = message_response

        result_standard = foundry_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])
        result_us = foundry_sync_provider.chat(
            "claude-sonnet-4-6",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(inference_geo="us"),
        )

        assert result_standard.cost is not None
        assert result_us.cost is not None
        assert result_us.cost.total_cost == result_standard.cost.total_cost

    def test_chat_exception_reports_foundry_provider(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_client: MagicMock,
        bad_request_error: anthropic.BadRequestError,
    ) -> None:
        mock_sync_client.messages.create.side_effect = bad_request_error

        with pytest.raises(InvalidRequestError) as exc_info:
            _ = foundry_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert exc_info.value.provider == "anthropic-foundry"


class TestFoundryAchat:
    async def test_basic_achat(
        self,
        foundry_async_provider: AnthropicFoundryProvider,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        result = await foundry_async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "anthropic-foundry"
        mock_async_client.messages.create.assert_awaited_once()


class TestFoundryChatStream:
    def test_stream_stamps_endpoint_identity(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        mock_sync_client.messages.create.return_value = iter(stream_events)

        chunks = list(foundry_sync_provider.chat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")]))

        assert chunks[-1].finish_reason == "stop"
        # The terminal chunk reports the Foundry endpoint identity, not the base provider name.
        assert chunks[-1].model == "claude-sonnet-4-6"
        assert chunks[-1].provider == "anthropic-foundry"

    async def test_async_stream_stamps_endpoint_identity(
        self,
        mock_async_foundry_create: MagicMock,
        mock_async_client: MagicMock,
        stream_events: list[MagicMock],
    ) -> None:
        assert mock_async_foundry_create  # fixture activates the patch
        token_auth = FakeFoundryTokenAuth()
        provider = AnthropicFoundryProvider(auth=token_auth, resource="my-resource")

        async def _async_iter() -> Any:  # noqa: ANN401
            for event in stream_events:
                yield event

        mock_async_client.messages.create.return_value = _async_iter()

        chunks = [chunk async for chunk in provider.achat_stream("claude-sonnet-4-6", [UserMessage(content="Hi")])]

        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].model == "claude-sonnet-4-6"
        assert chunks[-1].provider == "anthropic-foundry"


class TestFoundryClientManagement:
    def test_factory_receives_api_key_auth(
        self,
        foundry_sync_provider: AnthropicFoundryProvider,
        mock_sync_foundry_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response

        _ = foundry_sync_provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_foundry_create.assert_called_once_with(
            api_key="foundry-key",
            azure_ad_token_provider=None,
            resource="my-resource",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_factory_receives_token_provider_auth(
        self,
        mock_sync_foundry_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        token_auth = FakeFoundryTokenAuth()
        provider = AnthropicFoundryProvider(auth=token_auth, resource="my-resource")

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_foundry_create.assert_called_once_with(
            api_key=None,
            azure_ad_token_provider=token_auth.token_provider,
            resource="my-resource",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_factory_receives_overrides(
        self,
        fake_foundry_auth: FakeFoundryAuth,
        mock_sync_foundry_create: MagicMock,
        mock_sync_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_sync_client.messages.create.return_value = message_response
        provider = AnthropicFoundryProvider(
            auth=fake_foundry_auth,
            base_url="https://example-resource.services.ai.azure.com/anthropic/",
            timeout=30.0,
            max_retries=5,
        )

        _ = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_sync_foundry_create.assert_called_once_with(
            api_key="foundry-key",
            azure_ad_token_provider=None,
            resource=None,
            base_url="https://example-resource.services.ai.azure.com/anthropic/",
            timeout=30.0,
            max_retries=5,
        )

    async def test_async_factory_receives_api_key_auth(
        self,
        foundry_async_provider: AnthropicFoundryProvider,
        mock_async_foundry_create: MagicMock,
        mock_async_client: MagicMock,
        message_response: MagicMock,
    ) -> None:
        mock_async_client.messages.create.return_value = message_response

        _ = await foundry_async_provider.achat("claude-sonnet-4-6", [UserMessage(content="Hi")])

        mock_async_foundry_create.assert_called_once_with(
            api_key="foundry-key",
            azure_ad_token_provider=None,
            resource="my-resource",
            base_url=None,
            timeout=None,
            max_retries=None,
        )

    def test_default_auth_is_env(self) -> None:
        provider = AnthropicFoundryProvider()
        assert isinstance(provider._foundry_auth, AnthropicFoundryEnvAuthProvider)  # pyright: ignore[reportPrivateUsage]
