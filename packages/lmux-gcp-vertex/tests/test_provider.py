"""Tests for Google Vertex AI provider."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import ProviderError
from lmux.types import (
    ChatResponse,
    EmbeddingResponse,
    FunctionDefinition,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextResponseFormat,
    Tool,
    Usage,
    UserMessage,
)
from lmux_gcp_vertex import preload
from lmux_gcp_vertex.params import GCPVertexParams, SafetySetting
from lmux_gcp_vertex.provider import GCPVertexProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider for testing."""

    def get_credentials(self) -> MagicMock:
        return MagicMock()

    async def aget_credentials(self) -> MagicMock:
        return MagicMock()


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


def _make_response_mock(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int | None = None,
    finish_reason: str = "STOP",
) -> MagicMock:
    """Create a mock GenerateContentResponse."""
    response = MagicMock()
    part = MagicMock()
    part.thought = False
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = MagicMock(value=finish_reason)

    response.candidates = [candidate]

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = output_tokens
    usage.cached_content_token_count = cached_tokens
    response.usage_metadata = usage

    return response


def _make_embed_response_mock(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock EmbedContentResponse."""
    response = MagicMock()
    emb_mocks = []
    for values in embeddings:
        emb = MagicMock()
        emb.values = values
        emb_mocks.append(emb)
    response.embeddings = emb_mocks
    return response


@pytest.fixture
def generate_response() -> MagicMock:
    return _make_response_mock()


@pytest.fixture
def embed_response() -> MagicMock:
    return _make_embed_response_mock([[0.1, 0.2, 0.3]])


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_create(mock_client: MagicMock) -> Iterator[MagicMock]:
    with patch("lmux_gcp_vertex.provider.create_client", return_value=mock_client) as mock_create:
        yield mock_create


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_create: MagicMock) -> GCPVertexProvider:
    return GCPVertexProvider(auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_create: MagicMock) -> GCPVertexProvider:
    return GCPVertexProvider(auth=fake_auth)


@pytest.fixture
def server_error() -> Exception:
    return Exception("test error")


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        result = sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=result.cost,
            model="gemini-2.0-flash",
            provider="gcp-vertex",
            finish_reason="stop",
        )
        assert result.cost is not None
        assert result.cost.total_cost > 0
        mock_client.models.generate_content.assert_called_once()
        mock_client.models.embed_content.assert_not_called()

    def test_chat_with_params(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat(
            "gemini-2.0-flash",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config["temperature"] == 0.5
        assert config["max_output_tokens"] == 100
        assert config["top_p"] == 0.9
        assert config["stop_sequences"] == ["END"]

    def test_chat_with_stop_string(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], stop="STOP")

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["config"]["stop_sequences"] == ["STOP"]

    def test_chat_with_tools(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], tools=tools)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["config"]["tools"] == [{"function_declarations": [{"name": "get_weather"}]}]

    def test_chat_with_text_response_format(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        result = sync_provider.chat(
            "gemini-2.0-flash", [UserMessage(content="Hi")], response_format=TextResponseFormat()
        )

        assert result.content == "Hello!"
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert "response_mime_type" not in call_kwargs["config"]

    def test_chat_with_json_response_format(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat())

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["config"]["response_mime_type"] == "application/json"

    def test_chat_with_json_schema_response_format(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], response_format=rf)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["config"]["response_mime_type"] == "application/json"
        assert call_kwargs["config"]["response_schema"] == {"type": "object"}

    def test_chat_with_provider_params(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat(
            "gemini-2.0-flash",
            [UserMessage(content="Hi")],
            provider_params=GCPVertexParams(
                safety_settings=[SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")],
                presence_penalty=0.5,
            ),
        )

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config["safety_settings"] == [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ]
        assert config["presence_penalty"] == 0.5

    def test_chat_exception_mapping(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, server_error: Exception
    ) -> None:
        mock_client.models.generate_content.side_effect = server_error

        with pytest.raises(ProviderError):
            sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

    def test_chat_with_system_message(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat(
            "gemini-2.0-flash",
            [SystemMessage(content="Be helpful."), UserMessage(content="Hi")],
        )

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["config"]["system_instruction"] == "Be helpful."
        assert call_kwargs["contents"] == [{"role": "user", "parts": [{"text": "Hi"}]}]


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.aio.models.generate_content = AsyncMock(return_value=generate_response)

        result = await async_provider.achat("gemini-2.0-flash", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "gcp-vertex"
        mock_client.aio.models.generate_content.assert_awaited_once()
        mock_client.aio.models.embed_content.assert_not_called()

    async def test_achat_exception_mapping(
        self, async_provider: GCPVertexProvider, mock_client: MagicMock, server_error: Exception
    ) -> None:
        mock_client.aio.models.generate_content = AsyncMock(side_effect=server_error)

        with pytest.raises(ProviderError):
            await async_provider.achat("gemini-2.0-flash", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
    ) -> None:
        chunk1 = _make_response_mock(text="Hello")
        chunk1.usage_metadata = None
        chunk1.candidates[0].finish_reason = None

        chunk2 = _make_response_mock(text="")
        chunk2.candidates[0].content = None
        chunk2.candidates[0].finish_reason = MagicMock(value="STOP")
        chunk2.usage_metadata = None

        chunk3 = _make_response_mock()
        chunk3.candidates = None

        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2, chunk3])

        chunks = list(sync_provider.chat_stream("gemini-2.0-flash", [UserMessage(content="Hi")]))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert chunks[2].usage.input_tokens == 10

    def test_cost_on_usage_chunk(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
    ) -> None:
        chunk1 = _make_response_mock(text="Hello")
        chunk1.usage_metadata = None
        chunk1.candidates[0].finish_reason = None

        chunk2 = _make_response_mock()
        chunk2.candidates = None

        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        chunks = list(sync_provider.chat_stream("gemini-2.0-flash", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is not None
        assert chunks[1].cost.total_cost > 0

    def test_stream_exception_on_create(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, server_error: Exception
    ) -> None:
        mock_client.models.generate_content_stream.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("gemini-2.0-flash", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
        server_error: Exception,
    ) -> None:
        def _failing_iter() -> Any:  # noqa: ANN401
            yield _make_response_mock(text="Hi")
            raise server_error

        mock_client.models.generate_content_stream.return_value = _failing_iter()

        with pytest.raises(ProviderError, match="test error"):
            list(sync_provider.chat_stream("gemini-2.0-flash", [UserMessage(content="Hi")]))


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(
        self,
        async_provider: GCPVertexProvider,
        mock_client: MagicMock,
    ) -> None:
        chunk1 = _make_response_mock(text="Hello")
        chunk1.usage_metadata = None
        chunk1.candidates[0].finish_reason = None

        chunk2 = _make_response_mock()
        chunk2.candidates = None

        async def _async_iter() -> Any:  # noqa: ANN401
            yield chunk1
            yield chunk2

        mock_client.aio.models.generate_content_stream.return_value = _async_iter()

        chunks = [chunk async for chunk in async_provider.achat_stream("gemini-2.0-flash", [UserMessage(content="Hi")])]

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].cost is not None

    async def test_exception_during_stream(
        self,
        async_provider: GCPVertexProvider,
        mock_client: MagicMock,
        server_error: Exception,
    ) -> None:
        async def _failing_iter() -> Any:  # noqa: ANN401
            yield _make_response_mock(text="Hi")
            raise server_error

        mock_client.aio.models.generate_content_stream.return_value = _failing_iter()

        with pytest.raises(ProviderError, match="test error"):
            async for _ in async_provider.achat_stream("gemini-2.0-flash", [UserMessage(content="Hi")]):
                pass


# MARK: Embed


class TestEmbed:
    def test_basic_embed(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
        embed_response: MagicMock,
    ) -> None:
        mock_client.models.embed_content.return_value = embed_response

        result = sync_provider.embed("text-embedding-005", "hello")

        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=0, output_tokens=0),
            cost=result.cost,
            model="text-embedding-005",
            provider="gcp-vertex",
        )
        mock_client.models.embed_content.assert_called_once_with(model="text-embedding-005", contents=["hello"])
        mock_client.models.generate_content.assert_not_called()

    def test_embed_list_input(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
    ) -> None:
        mock_client.models.embed_content.return_value = _make_embed_response_mock([[0.1, 0.2], [0.3, 0.4]])

        result = sync_provider.embed("text-embedding-005", ["hello", "world"])

        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.models.embed_content.assert_called_once_with(
            model="text-embedding-005", contents=["hello", "world"]
        )

    def test_embed_exception_mapping(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, server_error: Exception
    ) -> None:
        mock_client.models.embed_content.side_effect = server_error

        with pytest.raises(ProviderError):
            sync_provider.embed("text-embedding-005", "hello")


# MARK: Aembed


class TestAembed:
    async def test_basic_aembed(
        self,
        async_provider: GCPVertexProvider,
        mock_client: MagicMock,
        embed_response: MagicMock,
    ) -> None:
        mock_client.aio.models.embed_content = AsyncMock(return_value=embed_response)

        result = await async_provider.aembed("text-embedding-005", "hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "gcp-vertex"
        mock_client.aio.models.embed_content.assert_awaited_once_with(model="text-embedding-005", contents=["hello"])
        mock_client.aio.models.generate_content.assert_not_called()

    async def test_aembed_list_input(
        self,
        async_provider: GCPVertexProvider,
        mock_client: MagicMock,
    ) -> None:
        mock_client.aio.models.embed_content = AsyncMock(return_value=_make_embed_response_mock([[0.1], [0.2]]))

        result = await async_provider.aembed("text-embedding-005", ["hello", "world"])

        assert result.embeddings == [[0.1], [0.2]]
        mock_client.aio.models.embed_content.assert_awaited_once_with(
            model="text-embedding-005", contents=["hello", "world"]
        )

    async def test_aembed_exception_mapping(
        self, async_provider: GCPVertexProvider, mock_client: MagicMock, server_error: Exception
    ) -> None:
        mock_client.aio.models.embed_content = AsyncMock(side_effect=server_error)

        with pytest.raises(ProviderError):
            await async_provider.aembed("text-embedding-005", "hello")


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self,
        sync_provider: GCPVertexProvider,
        mock_client: MagicMock,
        generate_response: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])
        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi again")])

        mock_create.assert_called_once()

    def test_project_and_location_passed(
        self,
        fake_auth: FakeAuth,
        mock_create: MagicMock,
        mock_client: MagicMock,
        generate_response: MagicMock,
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response
        provider = GCPVertexProvider(auth=fake_auth, project="my-project", location="us-central1")
        provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["location"] == "us-central1"
        assert call_kwargs["vertexai"] is True

    def test_api_key_passed(
        self,
        fake_auth: FakeAuth,
        mock_create: MagicMock,
        mock_client: MagicMock,
        generate_response: MagicMock,
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response
        provider = GCPVertexProvider(auth=fake_auth, vertexai=False, api_key="test-key")
        provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["vertexai"] is False

    async def test_async_client_reused(
        self,
        fake_auth: FakeAuth,
        mock_create: MagicMock,
        mock_client: MagicMock,
        generate_response: MagicMock,
    ) -> None:
        mock_client.aio.models.generate_content = AsyncMock(return_value=generate_response)
        provider = GCPVertexProvider(auth=fake_auth)
        await provider.achat("gemini-2.0-flash", [UserMessage(content="Hi")])
        await provider.achat("gemini-2.0-flash", [UserMessage(content="Hi again")])

        mock_create.assert_called_once()

    def test_default_options(
        self,
        fake_auth: FakeAuth,
        mock_create: MagicMock,
        mock_client: MagicMock,
        generate_response: MagicMock,
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response
        provider = GCPVertexProvider(auth=fake_auth)
        provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] is None
        assert call_kwargs["location"] is None
        assert call_kwargs["api_key"] is None


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(self, sync_provider: GCPVertexProvider, mock_client: MagicMock) -> None:
        custom_response = _make_response_mock(prompt_tokens=1000, output_tokens=500)
        mock_client.models.generate_content.return_value = custom_response

        sync_provider.register_pricing(
            "custom.my-model-v1",
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat("custom.my-model-v1", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        custom_pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
        )
        sync_provider.register_pricing("gemini-2.0-flash", custom_pricing)
        result = sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock
    ) -> None:
        unknown_response = _make_response_mock()
        mock_client.models.generate_content.return_value = unknown_response

        result = sync_provider.chat("totally-unknown-model", [UserMessage(content="Hi")])

        assert result.cost is None


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], provider_params=GCPVertexParams())

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert "safety_settings" not in config
        assert "presence_penalty" not in config
        assert "frequency_penalty" not in config
        assert "seed" not in config
        assert "labels" not in config
        assert "thinking_config" not in config

    def test_all_params(
        self, sync_provider: GCPVertexProvider, mock_client: MagicMock, generate_response: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = generate_response

        params = GCPVertexParams(
            safety_settings=[SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")],
            presence_penalty=0.5,
            frequency_penalty=0.3,
            seed=42,
            labels={"env": "test"},
            thinking_config={"thinking_budget": 1024},
        )
        sync_provider.chat("gemini-2.0-flash", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config["safety_settings"] == [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
        assert config["presence_penalty"] == 0.5
        assert config["frequency_penalty"] == 0.3
        assert config["seed"] == 42
        assert config["labels"] == {"env": "test"}
        assert config["thinking_config"] == {"thinking_budget": 1024}


# MARK: Preload


class TestPreload:
    def test_preload_imports_genai(self) -> None:
        preload()  # should not raise
