"""Tests for the Google provider (SDK-lite, respx)."""

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, NotFoundError, ProviderError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextResponseFormat,
    Tool,
    UserMessage,
)
from lmux_google import preload
from lmux_google.params import (
    DynamicRetrievalConfig,
    GoogleParams,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
)
from lmux_google.provider import GoogleProvider

_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-2.0-flash"
EMBED_MODEL = "text-embedding-005"
_CHAT_URL = f"{_GEMINI}/{MODEL}:generateContent"
_STREAM_URL = f"{_GEMINI}/{MODEL}:streamGenerateContent"
_EMBED_URL = f"{_GEMINI}/{EMBED_MODEL}:batchEmbedContents"


# MARK: Shared Fixtures


class FakeAPIKeyAuth:
    def get_credentials(self) -> str:
        return "test-api-key"

    async def aget_credentials(self) -> str:
        return "test-api-key"


class FakeCredentials:
    def __init__(self, *, valid: bool = True, token: str = "vertex-token", quota_project_id: str | None = None) -> None:  # noqa: S107
        self.valid = valid
        self.token = token
        self.quota_project_id = quota_project_id
        self.refreshed = False

    def refresh(self, _request: Any) -> None:  # noqa: ANN401
        self.refreshed = True
        self.valid = True
        self.token = "refreshed-token"  # noqa: S105


class FakeCredentialsAuth:
    def __init__(self, credentials: FakeCredentials) -> None:
        self._credentials = credentials

    def get_credentials(self) -> "Credentials":
        return cast("Credentials", self._credentials)

    async def aget_credentials(self) -> "Credentials":
        return cast("Credentials", self._credentials)


@pytest.fixture
def api_auth() -> FakeAPIKeyAuth:
    return FakeAPIKeyAuth()


@pytest.fixture
def provider(api_auth: FakeAPIKeyAuth) -> GoogleProvider:
    return GoogleProvider(auth=api_auth, vertexai=False)


@pytest.fixture
def sync_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_google.provider.create_sync_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_google.provider.create_async_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_two_clients(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    c1, c2 = MagicMock(), MagicMock()
    create = mocker.patch("lmux_google.provider.create_async_client", side_effect=[c1, c2])
    return create, c1, c2


@pytest.fixture
def mock_get_running_loop(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_google.provider.asyncio.get_running_loop")


def _gen_response(
    text: str = "Hello!", prompt_tokens: int = 10, output_tokens: int = 5, *, finish_reason: str = "STOP"
) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": output_tokens},
    }


@pytest.fixture
def gen_response() -> dict[str, Any]:
    return _gen_response()


def _sse_stream() -> bytes:
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "lo!"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    return ("\n\n".join(lines) + "\n\n").encode()


def _ok(respx_mock: respx.MockRouter, response: dict[str, Any], url: str = _CHAT_URL) -> respx.Route:
    return respx_mock.post(url).mock(return_value=httpx.Response(200, json=response))


# MARK: Chat


class TestChat:
    def test_basic(self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock, gen_response)
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.model == MODEL
        assert result.provider == "google"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.cost is not None
        assert result.cost.total_cost > 0
        assert route.called
        assert route.calls.last.request.headers["x-goog-api-key"] == "test-api-key"

    def test_request_body(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [SystemMessage(content="Be helpful."), UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "systemInstruction": {"parts": [{"text": "Be helpful."}]},
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 100,
                "topP": 0.9,
                "stopSequences": ["END"],
            },
        }

    def test_stop_string(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], stop="STOP")
        body = json.loads(route.calls.last.request.content)
        assert body["generationConfig"]["stopSequences"] == ["STOP"]

    def test_tools_and_choice(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
            tool_choice="required",
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"functionDeclarations": [{"name": "get_weather"}]}]
        assert body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}

    def test_text_response_format_no_mime(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], response_format=TextResponseFormat())
        body = json.loads(route.calls.last.request.content)
        assert "generationConfig" not in body

    def test_json_object_response_format(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat())
        body = json.loads(route.calls.last.request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "responseSchema" not in body["generationConfig"]

    def test_json_schema_response_format(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        rf = JsonSchemaResponseFormat(name="test", json_schema={"type": "object"})
        provider.chat(MODEL, [UserMessage(content="Hi")], response_format=rf)
        body = json.loads(route.calls.last.request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseJsonSchema"] == {"type": "object"}

    def test_reasoning_effort(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], reasoning_effort="medium")
        body = json.loads(route.calls.last.request.content)
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 8192, "includeThoughts": True}

    def test_status_error_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_non_json_body_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            provider.chat(MODEL, [UserMessage(content="Hi")])


# MARK: Achat


class TestAchat:
    async def test_basic(
        self, api_auth: FakeAPIKeyAuth, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        result = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "google"

    async def test_status_error_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(AuthenticationError):
            await provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_transport_error_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="refused"):
            await provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_non_json_body_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError):
            await provider.achat(MODEL, [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_and_costs(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[1].finish_reason == "stop"
        assert chunks[1].usage is not None
        assert chunks[0].cost is None
        assert chunks[1].cost is not None
        assert chunks[1].cost.total_cost > 0
        assert chunks[1].provider == "google"
        assert chunks[1].model == MODEL
        assert route.calls.last.request.url.params["alt"] == "sse"

    def test_status_error_on_open(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_malformed_chunk_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=b"data: {not json}\n\n"))
        with pytest.raises(ProviderError):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_mid_stream_error_raises_after_partial(
        self, provider: GoogleProvider, respx_mock: respx.MockRouter
    ) -> None:
        first = {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=sse))
        stream = provider.chat_stream(MODEL, [UserMessage(content="Hi")])
        assert next(stream).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            next(stream)

    def test_client_init_failure(self, api_auth: FakeAPIKeyAuth, sync_create_raises: MagicMock) -> None:
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="client init failed"):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        sync_create_raises.assert_called_once()


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_and_costs(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        chunks = [c async for c in provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[1].cost is not None
        assert chunks[1].provider == "google"
        assert chunks[1].model == MODEL

    async def test_status_error_on_open(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_chunk_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=b"data: {nope}\n\n"))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_mid_stream_error_raises_after_partial(
        self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter
    ) -> None:
        first = {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_STREAM_URL).mock(return_value=httpx.Response(200, content=sse))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        stream = provider.achat_stream(MODEL, [UserMessage(content="Hi")])
        assert (await anext(stream)).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            await anext(stream)

    async def test_client_init_failure(self, api_auth: FakeAPIKeyAuth, async_create_raises: MagicMock) -> None:
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="client init failed"):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover
        async_create_raises.assert_called_once()


# MARK: Embed


class TestEmbed:
    def test_basic(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]})
        )
        result = provider.embed(EMBED_MODEL, "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "google"
        body = json.loads(route.calls.last.request.content)
        assert body == {"requests": [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": "hello"}]}}]}

    def test_list_input(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]})
        )
        result = provider.embed(EMBED_MODEL, ["hello", "world"])
        assert result.embeddings == [[0.1], [0.2]]
        body = json.loads(route.calls.last.request.content)
        assert [r["content"]["parts"][0]["text"] for r in body["requests"]] == ["hello", "world"]

    def test_dimensions_and_task_type(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})
        )
        provider.embed(EMBED_MODEL, "hello", dimensions=256, provider_params=GoogleParams(task_type="RETRIEVAL_QUERY"))
        req = json.loads(route.calls.last.request.content)["requests"][0]
        assert req["outputDimensionality"] == 256
        assert req["taskType"] == "RETRIEVAL_QUERY"

    def test_exception_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            provider.embed(EMBED_MODEL, "hello")

    def test_status_error_mapped(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(return_value=httpx.Response(404, json={"error": {"message": "no"}}))
        with pytest.raises(NotFoundError):
            provider.embed(EMBED_MODEL, "hello")


# MARK: gemini-embedding-001 (one input per request)


_GEMINI_EMBED_001 = "gemini-embedding-001"


class TestGeminiEmbedding001Batching:
    # The whole gemini-embedding-* family is one-input-per-request; on the Developer API a multi-input
    # request would otherwise silently return a single aggregated embedding.
    @pytest.mark.parametrize("model", ["gemini-embedding-001", "gemini-embedding-2-preview"])
    def test_dev_api_splits_into_one_request_per_input(
        self, model: str, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter
    ) -> None:
        url = f"{_GEMINI}/{model}:batchEmbedContents"
        responses = [
            httpx.Response(200, json={"embeddings": [{"values": [0.1]}], "usageMetadata": {"promptTokenCount": 2}}),
            httpx.Response(200, json={"embeddings": [{"values": [0.2]}], "usageMetadata": {"promptTokenCount": 3}}),
        ]
        route = respx_mock.post(url).mock(side_effect=responses)
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        result = provider.embed(model, ["a", "b"])
        assert result.embeddings == [[0.1], [0.2]]
        assert result.usage.input_tokens == 5
        assert len(route.calls) == 2
        # Each request carries exactly one input.
        assert [len(json.loads(c.request.content)["requests"]) for c in route.calls] == [1, 1]

    def test_vertex_splits_into_one_request_per_input(self, respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{_GEMINI_EMBED_001}:predict"
        responses = [
            httpx.Response(200, json={"predictions": [{"embeddings": {"values": [0.1]}}]}),
            httpx.Response(200, json={"predictions": [{"embeddings": {"values": [0.2]}}]}),
        ]
        route = respx_mock.post(url).mock(side_effect=responses)
        result = provider.embed(_GEMINI_EMBED_001, ["a", "b"])
        assert result.embeddings == [[0.1], [0.2]]
        assert len(route.calls) == 2
        assert json.loads(route.calls[0].request.content) == {"instances": [{"content": "a"}]}

    async def test_aembed_splits_into_one_request_per_input(
        self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter
    ) -> None:
        url = f"{_GEMINI}/{_GEMINI_EMBED_001}:batchEmbedContents"
        responses = [
            httpx.Response(200, json={"embeddings": [{"values": [0.1]}]}),
            httpx.Response(200, json={"embeddings": [{"values": [0.2]}]}),
        ]
        route = respx_mock.post(url).mock(side_effect=responses)
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        result = await provider.aembed(_GEMINI_EMBED_001, ["a", "b"])
        assert result.embeddings == [[0.1], [0.2]]
        assert len(route.calls) == 2


# MARK: Aembed


class TestAembed:
    async def test_basic(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})
        )
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        result = await provider.aembed(EMBED_MODEL, "hello")
        assert result.embeddings == [[0.1, 0.2]]

    async def test_status_error_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(return_value=httpx.Response(404, json={"error": {"message": "no"}}))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(NotFoundError):
            await provider.aembed(EMBED_MODEL, "hello")

    async def test_transport_error_mapped(self, api_auth: FakeAPIKeyAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(side_effect=httpx.ConnectError("refused"))
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="refused"):
            await provider.aembed(EMBED_MODEL, "hello")


# MARK: Vertex Transport


class TestVertexTransport:
    def test_vertex_url_and_bearer(self, gen_response: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"

    def test_vertex_global_location(self, gen_response: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj")
        url = f"https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_vertex_refreshes_invalid_credentials(
        self, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        creds = FakeCredentials(valid=False)
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert creds.refreshed is True
        assert route.calls.last.request.headers["authorization"] == "Bearer refreshed-token"

    def test_vertex_requires_project(self) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds))
        with pytest.raises(ProviderError, match="requires a project"):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_vertex_reresolves_token_per_request(
        self, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls[0].request.headers["authorization"] == "Bearer vertex-token"
        # Credentials expire between requests — the second call must resolve a fresh token, not reuse baked headers.
        creds.valid = False
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls[1].request.headers["authorization"] == "Bearer refreshed-token"

    async def test_vertex_async_bearer(self, gen_response: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"

    def test_vertex_quota_project_header(self, gen_response: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials(quota_project_id="quota-proj")
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=gen_response))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls.last.request.headers["x-goog-user-project"] == "quota-proj"


# MARK: Vertex Embeddings


class TestVertexEmbed:
    def test_predict_endpoint_and_shape(self, respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{EMBED_MODEL}:predict"
        predictions = {"predictions": [{"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 3}}}]}
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=predictions))
        result = provider.embed(EMBED_MODEL, "hello")
        assert result.embeddings == [[0.1, 0.2]]
        assert result.usage.input_tokens == 3
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"
        body = json.loads(route.calls.last.request.content)
        assert body == {"instances": [{"content": "hello"}]}

    def test_dimensions_and_task_type(self, respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{EMBED_MODEL}:predict"
        predictions = {"predictions": [{"embeddings": {"values": [0.1]}}, {"embeddings": {"values": [0.2]}}]}
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=predictions))
        result = provider.embed(
            EMBED_MODEL, ["a", "b"], dimensions=256, provider_params=GoogleParams(task_type="RETRIEVAL_QUERY")
        )
        assert result.embeddings == [[0.1], [0.2]]
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "instances": [
                {"content": "a", "task_type": "RETRIEVAL_QUERY"},
                {"content": "b", "task_type": "RETRIEVAL_QUERY"},
            ],
            "parameters": {"outputDimensionality": 256},
        }

    async def test_aembed_predict(self, respx_mock: respx.MockRouter) -> None:
        creds = FakeCredentials()
        provider = GoogleProvider(auth=FakeCredentialsAuth(creds), project="my-proj", location="us-central1")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/{EMBED_MODEL}:predict"
        predictions = {"predictions": [{"embeddings": {"values": [0.3]}}]}
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=predictions))
        result = await provider.aembed(EMBED_MODEL, "hello")
        assert result.embeddings == [[0.3]]
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"


# MARK: Vertex embedContent (Gemini Embedding 2)


_EMBED2_MODEL = "gemini-embedding-2-preview"


def _embed_content_url(model: str = _EMBED2_MODEL) -> str:
    base = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models"
    return f"{base}/{model}:embedContent"


def _vertex_embed2_provider() -> GoogleProvider:
    return GoogleProvider(auth=FakeCredentialsAuth(FakeCredentials()), project="my-proj", location="us-central1")


class TestVertexEmbedContent:
    def test_single_input_endpoint_and_shape(self, respx_mock: respx.MockRouter) -> None:
        body = {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 4}}
        route = respx_mock.post(_embed_content_url()).mock(return_value=httpx.Response(200, json=body))
        result = _vertex_embed2_provider().embed(_EMBED2_MODEL, "hello")
        assert result.embeddings == [[0.1, 0.2]]
        assert result.usage.input_tokens == 4
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"
        # embedContent is single-content and rejects task_type — the body carries neither instances nor task_type.
        assert json.loads(route.calls.last.request.content) == {"content": {"parts": [{"text": "hello"}]}}

    def test_list_issues_one_request_per_item(self, respx_mock: respx.MockRouter) -> None:
        responses = [
            httpx.Response(200, json={"embedding": {"values": [0.1]}, "usageMetadata": {"promptTokenCount": 2}}),
            httpx.Response(200, json={"embedding": {"values": [0.2]}, "usageMetadata": {"promptTokenCount": 3}}),
        ]
        route = respx_mock.post(_embed_content_url()).mock(side_effect=responses)
        result = _vertex_embed2_provider().embed(_EMBED2_MODEL, ["a", "b"])
        assert result.embeddings == [[0.1], [0.2]]
        assert result.usage.input_tokens == 5
        assert len(route.calls) == 2
        assert json.loads(route.calls[0].request.content) == {"content": {"parts": [{"text": "a"}]}}

    def test_dimensions_forwarded(self, respx_mock: respx.MockRouter) -> None:
        body = {"embedding": {"values": [0.1]}}
        route = respx_mock.post(_embed_content_url()).mock(return_value=httpx.Response(200, json=body))
        _vertex_embed2_provider().embed(_EMBED2_MODEL, "hi", dimensions=128)
        assert json.loads(route.calls.last.request.content) == {
            "content": {"parts": [{"text": "hi"}]},
            "outputDimensionality": 128,
        }

    async def test_aembed_content(self, respx_mock: respx.MockRouter) -> None:
        body = {"embedding": {"values": [0.5]}, "usageMetadata": {"promptTokenCount": 1}}
        route = respx_mock.post(_embed_content_url()).mock(return_value=httpx.Response(200, json=body))
        result = await _vertex_embed2_provider().aembed(_EMBED2_MODEL, "hi")
        assert result.embeddings == [[0.5]]
        assert result.usage.input_tokens == 1
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"

    def test_status_error_mapped(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_embed_content_url()).mock(return_value=httpx.Response(404, json={"error": {"message": "no"}}))
        with pytest.raises(NotFoundError):
            _vertex_embed2_provider().embed(_EMBED2_MODEL, "hi")

    def test_transport_error_mapped(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_embed_content_url()).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            _vertex_embed2_provider().embed(_EMBED2_MODEL, "hi")

    async def test_aembed_transport_error_mapped(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_embed_content_url()).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await _vertex_embed2_provider().aembed(_EMBED2_MODEL, "hi")

    async def test_aembed_status_error_mapped(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_embed_content_url()).mock(return_value=httpx.Response(404, json={"error": {"message": "no"}}))
        with pytest.raises(NotFoundError):
            await _vertex_embed2_provider().aembed(_EMBED2_MODEL, "hi")

    async def test_aembed_list_issues_one_request_per_item(self, respx_mock: respx.MockRouter) -> None:
        responses = [
            httpx.Response(200, json={"embedding": {"values": [0.1]}, "usageMetadata": {"promptTokenCount": 2}}),
            httpx.Response(200, json={"embedding": {"values": [0.2]}, "usageMetadata": {"promptTokenCount": 3}}),
        ]
        route = respx_mock.post(_embed_content_url()).mock(side_effect=responses)
        result = await _vertex_embed2_provider().aembed(_EMBED2_MODEL, ["a", "b"])
        assert result.embeddings == [[0.1], [0.2]]
        assert result.usage.input_tokens == 5
        assert len(route.calls) == 2


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="a")])
        client = provider._sync_client
        provider.chat(MODEL, [UserMessage(content="b")])
        assert provider._sync_client is client

    async def test_async_client_reused(
        self, api_auth: FakeAPIKeyAuth, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        await provider.achat(MODEL, [UserMessage(content="a")])
        client = provider._async_client
        await provider.achat(MODEL, [UserMessage(content="b")])
        assert provider._async_client is client

    def test_timeout_passed(
        self, api_auth: FakeAPIKeyAuth, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider = GoogleProvider(auth=api_auth, vertexai=False, timeout=30.0, max_retries=5)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert provider._sync_client is not None
        assert provider._sync_client.timeout.read == 30.0

    def test_sync_init_failure_mapped(self, api_auth: FakeAPIKeyAuth, sync_create_raises: MagicMock) -> None:
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.chat(MODEL, [UserMessage(content="Hi")])
        sync_create_raises.assert_called_once()

    async def test_async_init_failure_mapped(self, api_auth: FakeAPIKeyAuth, async_create_raises: MagicMock) -> None:
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        with pytest.raises(ProviderError, match="client init failed"):
            await provider.achat(MODEL, [UserMessage(content="Hi")])
        async_create_raises.assert_called_once()

    async def test_async_client_recreated_on_new_loop(
        self,
        api_auth: FakeAPIKeyAuth,
        async_create_two_clients: tuple[MagicMock, MagicMock, MagicMock],
        mock_get_running_loop: MagicMock,
    ) -> None:
        create, c1, c2 = async_create_two_clients
        loop1, loop2 = asyncio.new_event_loop(), asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_for_unknown_model(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        model = "custom.my-model-v1"
        _ok(respx_mock, _gen_response(prompt_tokens=1000, output_tokens=500), url=f"{_GEMINI}/{model}:generateContent")
        provider.register_pricing(
            model, ModelPricing(tiers=[PricingTier(input_cost_per_token=5e-6, output_cost_per_token=15e-6)])
        )
        result = provider.chat(model, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5e-6)

    def test_custom_overrides_builtin(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider.register_pricing(
            MODEL, ModelPricing(tiers=[PricingTier(input_cost_per_token=99e-6, output_cost_per_token=199e-6)])
        )
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99e-6)

    def test_unknown_model_none_cost(self, provider: GoogleProvider, respx_mock: respx.MockRouter) -> None:
        model = "totally-unknown-model"
        _ok(respx_mock, _gen_response(), url=f"{_GEMINI}/{model}:generateContent")
        result = provider.chat(model, [UserMessage(content="Hi")])
        assert result.cost is None


# MARK: Provider Params


class TestProviderParams:
    def test_empty_params(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GoogleParams())
        body = json.loads(route.calls.last.request.content)
        assert "safetySettings" not in body
        assert "generationConfig" not in body
        assert "tools" not in body

    def test_all_generation_params(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                safety_settings=[SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")],
                presence_penalty=0.5,
                frequency_penalty=0.3,
                seed=42,
                labels={"env": "test"},
                thinking_config={"thinkingBudget": 1024},
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["safetySettings"] == [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
        assert body["labels"] == {"env": "test"}
        assert body["generationConfig"] == {
            "presencePenalty": 0.5,
            "frequencyPenalty": 0.3,
            "seed": 42,
            "thinkingConfig": {"thinkingBudget": 1024},
        }

    def test_google_search_bool(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GoogleParams(google_search=True))
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearch": {}}]

    def test_google_search_false_is_noop(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GoogleParams(google_search=False))
        body = json.loads(route.calls.last.request.content)
        assert "tools" not in body

    def test_google_search_config_full(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search=GoogleSearchConfig(
                    search_types=GoogleSearchTypes(web_search=True), exclude_domains=["example.com"]
                )
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [
            {"googleSearch": {"searchTypes": {"webSearch": {}}, "excludeDomains": ["example.com"]}}
        ]

    def test_google_search_config_image_search(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search=GoogleSearchConfig(search_types=GoogleSearchTypes(image_search=True))
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearch": {"searchTypes": {"imageSearch": {}}}}]

    def test_google_search_config_empty(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL, [UserMessage(content="Hi")], provider_params=GoogleParams(google_search=GoogleSearchConfig())
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearch": {}}]

    def test_google_search_config_empty_search_types(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(google_search=GoogleSearchConfig(search_types=GoogleSearchTypes())),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearch": {}}]

    def test_code_execution(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GoogleParams(code_execution=True))
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"codeExecution": {}}]

    def test_google_search_retrieval_full(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search_retrieval=GoogleSearchRetrievalConfig(
                    dynamic_retrieval_config=DynamicRetrievalConfig(mode="MODE_DYNAMIC", dynamic_threshold=0.5)
                )
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [
            {"googleSearchRetrieval": {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.5}}}
        ]

    def test_google_search_retrieval_mode_only(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search_retrieval=GoogleSearchRetrievalConfig(
                    dynamic_retrieval_config=DynamicRetrievalConfig(mode="MODE_DYNAMIC")
                )
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearchRetrieval": {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC"}}}]

    def test_google_search_retrieval_threshold_only(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search_retrieval=GoogleSearchRetrievalConfig(
                    dynamic_retrieval_config=DynamicRetrievalConfig(dynamic_threshold=0.7)
                )
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearchRetrieval": {"dynamicRetrievalConfig": {"dynamicThreshold": 0.7}}}]

    def test_google_search_retrieval_empty_drc(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(
                google_search_retrieval=GoogleSearchRetrievalConfig(dynamic_retrieval_config=DynamicRetrievalConfig())
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearchRetrieval": {}}]

    def test_google_search_retrieval_no_drc(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=GoogleParams(google_search_retrieval=GoogleSearchRetrievalConfig()),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"googleSearchRetrieval": {}}]

    def test_special_tools_merge_with_function_tools(
        self, provider: GoogleProvider, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock, gen_response)
        provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
            provider_params=GoogleParams(google_search=True, code_execution=True),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [
            {"functionDeclarations": [{"name": "get_weather"}]},
            {"googleSearch": {}},
            {"codeExecution": {}},
        ]


# MARK: Aclose & Preload


class TestAclose:
    async def test_closes_client(
        self, api_auth: FakeAPIKeyAuth, gen_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock, gen_response)
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert provider._async_client is not None
        await provider.aclose()
        assert provider._async_client is None

    async def test_noop_when_no_client(self, api_auth: FakeAPIKeyAuth) -> None:
        provider = GoogleProvider(auth=api_auth, vertexai=False)
        await provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()
