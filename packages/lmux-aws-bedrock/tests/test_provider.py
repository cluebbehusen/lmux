"""Tests for AWS Bedrock provider."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import ProviderError, UnsupportedFeatureError
from lmux.types import (
    ChatChunk,
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
from lmux_aws_bedrock import preload
from lmux_aws_bedrock.params import BedrockParams, GuardrailConfig
from lmux_aws_bedrock.provider import BedrockProvider

# MARK: Shared Fixtures


class FakeAuth:
    """Fake auth provider for testing."""

    def __init__(self, async_session: MagicMock | None = None) -> None:
        self.async_session: MagicMock = async_session or MagicMock()
        self.get_credentials_calls: int = 0
        self.aget_credentials_calls: int = 0

    def get_credentials(self) -> MagicMock:
        self.get_credentials_calls += 1
        return MagicMock()

    async def aget_credentials(self) -> MagicMock:
        self.aget_credentials_calls += 1
        return self.async_session


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def converse_response() -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


@pytest.fixture
def stream_events() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}, "metrics": {"latencyMs": 100}}},
    ]


def _make_embedding_body(embedding: list[float], token_count: int) -> MagicMock:
    """Create a mock streaming body for invoke_model embedding responses."""
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": embedding, "inputTextTokenCount": token_count}).encode()
    return body


@pytest.fixture
def embedding_invoke_response() -> dict[str, Any]:
    return {"body": _make_embedding_body([0.1, 0.2, 0.3], 5)}


@pytest.fixture
def mock_sync_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_sync_create(mock_sync_client: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_sync_client", return_value=mock_sync_client)


@pytest.fixture
def sync_provider(fake_auth: FakeAuth, mock_sync_create: MagicMock) -> BedrockProvider:
    provider = BedrockProvider(auth=fake_auth)
    mock_sync_create.assert_not_called()  # client created lazily, not at init
    return provider


@pytest.fixture
def mock_async_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_async_client_ctx(mock_async_client: AsyncMock) -> AsyncMock:
    mock_ctx_manager = AsyncMock()
    mock_ctx_manager.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_ctx_manager.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx_manager


@pytest.fixture
def mock_async_create(mock_async_client_ctx: AsyncMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_async_client", return_value=mock_async_client_ctx)


@pytest.fixture
def async_provider(fake_auth: FakeAuth, mock_async_create: MagicMock) -> BedrockProvider:
    provider = BedrockProvider(auth=fake_auth)
    mock_async_create.assert_not_called()  # client created lazily, not at init
    return provider


@pytest.fixture
def server_error() -> Exception:
    """Create a generic exception for server-side errors."""
    return Exception("test error")


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        result = sync_provider.chat("amazon.nova-pro-v1", [UserMessage(content="Hi")])

        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=result.cost,  # Cost is calculated, just verify it's present
            model="amazon.nova-pro-v1",
            provider="aws-bedrock",
            finish_reason="stop",
        )
        assert result.cost is not None
        assert result.cost.total_cost > 0
        mock_sync_client.converse.assert_called_once()
        mock_sync_client.invoke_model.assert_not_called()

    def test_chat_with_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )

        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"temperature": 0.5, "maxTokens": 100, "topP": 0.9, "stopSequences": ["END"]},
        )

    def test_chat_with_tools(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        tools = [Tool(function=FunctionDefinition(name="get_weather"))]
        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")], tools=tools)

        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            toolConfig={"tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {"json": {"type": "object"}}}}]},
        )

    def test_chat_with_text_response_format(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        result = sync_provider.chat(
            "anthropic.claude-sonnet-4", [UserMessage(content="Hi")], response_format=TextResponseFormat()
        )

        assert result.content == "Hello!"
        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

    def test_chat_json_object_raises(self, sync_provider: BedrockProvider) -> None:
        with pytest.raises(UnsupportedFeatureError, match="JsonObjectResponseFormat is not supported"):
            sync_provider.chat(
                "anthropic.claude-sonnet-4", [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat()
            )

    def test_chat_with_json_schema_response_format(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            response_format=JsonSchemaResponseFormat(
                name="weather_response",
                description="Structured weather payload",
                json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )

        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            outputConfig={
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(
                                {
                                    "additionalProperties": False,
                                    "properties": {"city": {"type": "string"}},
                                    "type": "object",
                                },
                                sort_keys=True,
                            ),
                            "name": "weather_response",
                            "description": "Structured weather payload",
                        }
                    },
                }
            },
        )

    def test_chat_with_provider_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            provider_params=BedrockParams(
                guardrail_config=GuardrailConfig(guardrail_identifier="g1", guardrail_version="1"),
            ),
        )

        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            guardrailConfig={"guardrailIdentifier": "g1", "guardrailVersion": "1"},
        )

    def test_chat_exception_mapping(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, server_error: Exception
    ) -> None:
        mock_sync_client.converse.side_effect = server_error

        with pytest.raises(ProviderError):
            sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

    def test_chat_with_system_message(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [SystemMessage(content="Be helpful."), UserMessage(content="Hi")],
        )

        mock_sync_client.converse.assert_called_once_with(
            modelId="anthropic.claude-sonnet-4",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            system=[{"text": "Be helpful."}],
        )

    def test_chat_with_stop_string(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")], stop="STOP")

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        assert call_kwargs["inferenceConfig"]["stopSequences"] == ["STOP"]

    def test_chat_with_reasoning_effort(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")], reasoning_effort="medium")

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        assert call_kwargs["additionalModelRequestFields"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 8192,
        }

    def test_chat_reasoning_effort_deep_merges_with_provider_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=BedrockParams(additional_model_request_fields={"some_field": "value"}),
        )

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        additional = call_kwargs["additionalModelRequestFields"]
        assert additional["thinking"] == {"type": "enabled", "budget_tokens": 32768}
        assert additional["some_field"] == "value"

    def test_chat_provider_params_thinking_overrides_reasoning_effort(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=BedrockParams(
                additional_model_request_fields={"thinking": {"type": "enabled", "budget_tokens": 99999}}
            ),
        )

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        assert call_kwargs["additionalModelRequestFields"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 99999,
        }

    def test_chat_reasoning_effort_does_not_mutate_provider_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response
        fields: dict[str, Any] = {"some_field": "value"}
        params = BedrockParams(additional_model_request_fields=fields)

        sync_provider.chat(
            "anthropic.claude-sonnet-4",
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=params,
        )

        # The original dict must not have been mutated
        assert "thinking" not in fields


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: BedrockProvider, mock_async_client: AsyncMock, converse_response: dict[str, Any]
    ) -> None:
        mock_async_client.converse.return_value = converse_response

        result = await async_provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        assert result.content == "Hello!"
        assert result.provider == "aws-bedrock"
        mock_async_client.converse.assert_awaited_once()
        mock_async_client.invoke_model.assert_not_called()

    async def test_achat_exception_mapping(
        self, async_provider: BedrockProvider, mock_async_client: AsyncMock, server_error: Exception
    ) -> None:
        mock_async_client.converse.side_effect = server_error

        with pytest.raises(ProviderError):
            await async_provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
        stream_events: list[dict[str, Any]],
    ) -> None:
        mock_sync_client.converse_stream.return_value = {"stream": iter(stream_events)}

        chunks = list(sync_provider.chat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]))

        # messageStart is skipped (returns None), contentBlockDelta yields text,
        # messageStop yields finish_reason, metadata yields usage
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert chunks[2].usage.input_tokens == 10
        assert chunks[2].usage.output_tokens == 5

    def test_cost_on_metadata_chunk(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
        stream_events: list[dict[str, Any]],
    ) -> None:
        mock_sync_client.converse_stream.return_value = {"stream": iter(stream_events)}

        chunks = list(sync_provider.chat_stream("amazon.nova-pro-v1", [UserMessage(content="Hi")]))

        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_stream_exception_on_create(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, server_error: Exception
    ) -> None:
        mock_sync_client.converse_stream.side_effect = server_error

        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]))

    def test_stream_exception_during_iteration(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
        stream_events: list[dict[str, Any]],
        server_error: Exception,
    ) -> None:
        def _failing_iter() -> Any:  # noqa: ANN401
            yield stream_events[1]  # contentBlockDelta
            raise server_error

        mock_sync_client.converse_stream.return_value = {"stream": _failing_iter()}

        with pytest.raises(ProviderError, match="test error"):
            list(sync_provider.chat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]))

    def test_stream_chunk_without_usage_has_no_cost(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
    ) -> None:
        """Chunks that have no usage (non-metadata chunks) should have cost=None."""
        events = [
            {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        mock_sync_client.converse_stream.return_value = {"stream": iter(events)}

        chunks = list(sync_provider.chat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]))

        assert chunks[0] == ChatChunk(delta="Hi")
        assert chunks[0].cost is None
        assert chunks[1].cost is None


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(
        self,
        async_provider: BedrockProvider,
        mock_async_client: AsyncMock,
        stream_events: list[dict[str, Any]],
    ) -> None:
        async def _async_iter() -> Any:  # noqa: ANN401
            for event in stream_events:
                yield event

        mock_async_client.converse_stream.return_value = {"stream": _async_iter()}

        chunks = [
            chunk async for chunk in async_provider.achat_stream("amazon.nova-pro-v1", [UserMessage(content="Hi")])
        ]

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[2].cost is not None

    async def test_exception_on_create(
        self, async_provider: BedrockProvider, mock_async_client: AsyncMock, server_error: Exception
    ) -> None:
        mock_async_client.converse_stream.side_effect = server_error

        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_during_iteration(
        self,
        async_provider: BedrockProvider,
        mock_async_client: AsyncMock,
        stream_events: list[dict[str, Any]],
        server_error: Exception,
    ) -> None:
        async def _failing_async_iter() -> Any:  # noqa: ANN401
            yield stream_events[1]  # contentBlockDelta
            raise server_error

        mock_async_client.converse_stream.return_value = {"stream": _failing_async_iter()}

        with pytest.raises(ProviderError, match="test error"):
            async for _ in async_provider.achat_stream("anthropic.claude-sonnet-4", [UserMessage(content="Hi")]):
                pass


# MARK: Embed


class TestEmbed:
    def test_basic_embed(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
        embedding_invoke_response: dict[str, Any],
    ) -> None:
        mock_sync_client.invoke_model.return_value = embedding_invoke_response

        result = sync_provider.embed("amazon.titan-embed-text-v2", "hello")

        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=result.cost,
            model="amazon.titan-embed-text-v2",
            provider="aws-bedrock",
        )
        mock_sync_client.invoke_model.assert_called_once_with(
            modelId="amazon.titan-embed-text-v2",
            contentType="application/json",
            body=json.dumps({"inputText": "hello"}),
        )
        mock_sync_client.converse.assert_not_called()

    def test_embed_list_input(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
    ) -> None:
        mock_sync_client.invoke_model.side_effect = [
            {"body": _make_embedding_body([0.1, 0.2, 0.3], 3)},
            {"body": _make_embedding_body([0.4, 0.5, 0.6], 4)},
        ]

        result = sync_provider.embed("amazon.titan-embed-text-v2", ["hello", "world"])

        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            usage=Usage(input_tokens=7, output_tokens=0),
            cost=result.cost,
            model="amazon.titan-embed-text-v2",
            provider="aws-bedrock",
        )
        assert mock_sync_client.invoke_model.call_count == 2

    def test_embed_with_dimensions(
        self,
        sync_provider: BedrockProvider,
        mock_sync_client: MagicMock,
        embedding_invoke_response: dict[str, Any],
    ) -> None:
        mock_sync_client.invoke_model.return_value = embedding_invoke_response

        sync_provider.embed("amazon.titan-embed-text-v2", "hello", dimensions=256)

        mock_sync_client.invoke_model.assert_called_once_with(
            modelId="amazon.titan-embed-text-v2",
            contentType="application/json",
            body=json.dumps({"inputText": "hello", "dimensions": 256}),
        )

    def test_embed_exception_mapping(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, server_error: Exception
    ) -> None:
        mock_sync_client.invoke_model.side_effect = server_error

        with pytest.raises(ProviderError):
            sync_provider.embed("amazon.titan-embed-text-v2", "hello")

    def test_embed_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
    ) -> None:
        mock_sync_create.side_effect = Exception("connection refused")
        provider = BedrockProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            provider.embed("amazon.titan-embed-text-v2", "hello")


# MARK: Aembed


class TestAembed:
    async def test_basic_aembed(
        self,
        async_provider: BedrockProvider,
        mock_async_client: AsyncMock,
    ) -> None:
        mock_body = AsyncMock()
        mock_body.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}).encode()
        mock_async_client.invoke_model.return_value = {"body": mock_body}

        result = await async_provider.aembed("amazon.titan-embed-text-v2", "hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.usage == Usage(input_tokens=5, output_tokens=0)
        assert result.provider == "aws-bedrock"
        mock_async_client.invoke_model.assert_awaited_once_with(
            modelId="amazon.titan-embed-text-v2",
            contentType="application/json",
            body=json.dumps({"inputText": "hello"}),
        )
        mock_async_client.converse.assert_not_called()

    async def test_aembed_list_input(
        self,
        async_provider: BedrockProvider,
        mock_async_client: AsyncMock,
    ) -> None:
        mock_body1 = AsyncMock()
        mock_body1.read.return_value = json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 3}).encode()
        mock_body2 = AsyncMock()
        mock_body2.read.return_value = json.dumps({"embedding": [0.3, 0.4], "inputTextTokenCount": 4}).encode()
        mock_async_client.invoke_model.side_effect = [{"body": mock_body1}, {"body": mock_body2}]

        result = await async_provider.aembed("amazon.titan-embed-text-v2", ["hello", "world"])

        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.usage == Usage(input_tokens=7, output_tokens=0)
        assert mock_async_client.invoke_model.call_count == 2

    async def test_aembed_with_dimensions(
        self,
        async_provider: BedrockProvider,
        mock_async_client: AsyncMock,
    ) -> None:
        mock_body = AsyncMock()
        mock_body.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}).encode()
        mock_async_client.invoke_model.return_value = {"body": mock_body}

        await async_provider.aembed("amazon.titan-embed-text-v2", "hello", dimensions=256)

        mock_async_client.invoke_model.assert_awaited_once_with(
            modelId="amazon.titan-embed-text-v2",
            contentType="application/json",
            body=json.dumps({"inputText": "hello", "dimensions": 256}),
        )

    async def test_aembed_exception_mapping(
        self, async_provider: BedrockProvider, mock_async_client: AsyncMock, server_error: Exception
    ) -> None:
        mock_async_client.invoke_model.side_effect = server_error

        with pytest.raises(ProviderError):
            await async_provider.aembed("amazon.titan-embed-text-v2", "hello")


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])
        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi again")])

        assert mock_sync_client.converse.call_count == 2

    def test_region_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_sync_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth, region="us-west-2")
        provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once()
        call_kwargs = mock_sync_create.call_args.kwargs
        assert call_kwargs["region_name"] == "us-west-2"

    def test_endpoint_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_sync_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth, endpoint_url="https://custom.bedrock.endpoint")
        provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once()
        call_kwargs = mock_sync_create.call_args.kwargs
        assert call_kwargs["endpoint_url"] == "https://custom.bedrock.endpoint"

    def test_create_sync_client_called_once(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_sync_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth)
        provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])
        provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi again")])

        mock_sync_create.assert_called_once()

    async def test_async_session_reused(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: AsyncMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_async_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth)
        await provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])
        await provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi again")])

        assert fake_auth.aget_credentials_calls == 1
        assert mock_async_create.call_count == 2

    async def test_async_region_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: AsyncMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_async_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth, region="eu-west-1")
        await provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(fake_auth.async_session, region_name="eu-west-1", endpoint_url=None)

    async def test_async_endpoint_url_passed(
        self,
        fake_auth: FakeAuth,
        mock_async_create: MagicMock,
        mock_async_client: AsyncMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_async_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth, endpoint_url="https://custom.endpoint")
        await provider.achat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        mock_async_create.assert_called_once_with(
            fake_auth.async_session,
            region_name=None,
            endpoint_url="https://custom.endpoint",
        )

    def test_sync_client_init_failure_mapped(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
    ) -> None:
        mock_sync_create.side_effect = Exception("connection refused")
        provider = BedrockProvider(auth=fake_auth)

        with pytest.raises(ProviderError, match="connection refused"):
            provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

    def test_no_region_no_endpoint_defaults_to_none(
        self,
        fake_auth: FakeAuth,
        mock_sync_create: MagicMock,
        mock_sync_client: MagicMock,
        converse_response: dict[str, Any],
    ) -> None:
        mock_sync_client.converse.return_value = converse_response
        provider = BedrockProvider(auth=fake_auth)
        provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        mock_sync_create.assert_called_once()
        call_kwargs = mock_sync_create.call_args.kwargs
        assert call_kwargs["region_name"] is None
        assert call_kwargs["endpoint_url"] is None


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock
    ) -> None:
        """Custom pricing populates cost for models not in built-in PRICING."""
        custom_response: dict[str, Any] = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
        }
        mock_sync_client.converse.return_value = custom_response

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
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        """User-registered pricing takes precedence over built-in pricing."""
        mock_sync_client.converse.return_value = converse_response

        custom_pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
        )
        sync_provider.register_pricing("anthropic.claude-sonnet-4", custom_pricing)
        result = sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")])

        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock
    ) -> None:
        """Without registration, unknown models return cost=None."""
        unknown_response: dict[str, Any] = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        mock_sync_client.converse.return_value = unknown_response

        result = sync_provider.chat("totally-unknown-model", [UserMessage(content="Hi")])

        assert result.cost is None


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")], provider_params=BedrockParams())

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        assert "guardrailConfig" not in call_kwargs
        assert "additionalModelRequestFields" not in call_kwargs
        assert "additionalModelResponseFieldPaths" not in call_kwargs

    def test_all_params(
        self, sync_provider: BedrockProvider, mock_sync_client: MagicMock, converse_response: dict[str, Any]
    ) -> None:
        mock_sync_client.converse.return_value = converse_response

        params = BedrockParams(
            guardrail_config=GuardrailConfig(guardrail_identifier="g1", guardrail_version="1", trace="enabled"),
            additional_model_request_fields={"custom_field": "value"},
            additional_model_response_field_paths=["$.path.to.field"],
        )
        sync_provider.chat("anthropic.claude-sonnet-4", [UserMessage(content="Hi")], provider_params=params)

        call_kwargs = mock_sync_client.converse.call_args.kwargs
        assert call_kwargs["guardrailConfig"] == {
            "guardrailIdentifier": "g1",
            "guardrailVersion": "1",
            "trace": "enabled",
        }
        assert call_kwargs["additionalModelRequestFields"] == {"custom_field": "value"}
        assert call_kwargs["additionalModelResponseFieldPaths"] == ["$.path.to.field"]


# MARK: Preload


class TestPreload:
    @pytest.fixture
    def mock_missing_aiobotocore(self, mocker: MockerFixture) -> None:
        mocker.patch.dict("sys.modules", {"aiobotocore": None})

    def test_preload_imports_boto3_and_aiobotocore(self) -> None:
        preload()  # should not raise; aiobotocore is installed in dev

    def test_preload_without_aiobotocore(self, mock_missing_aiobotocore: None) -> None:
        assert mock_missing_aiobotocore is None  # side-effect fixture: patches sys.modules
        preload()  # should not raise when aiobotocore is missing
