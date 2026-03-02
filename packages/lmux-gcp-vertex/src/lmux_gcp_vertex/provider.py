"""GCP Vertex AI provider implementation."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai import Client

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, EmbeddingProvider, PricingProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    Tool,
    Usage,
)
from lmux_gcp_vertex._exceptions import map_gcp_vertex_error
from lmux_gcp_vertex._lazy import create_client
from lmux_gcp_vertex._mappers import (
    map_embed_content_response,
    map_generate_content_chunk,
    map_generate_content_response,
    map_messages,
    map_response_format,
    map_tools,
)
from lmux_gcp_vertex.auth import GCPVertexADCAuthProvider
from lmux_gcp_vertex.cost import calculate_gcp_vertex_cost
from lmux_gcp_vertex.params import GCPVertexParams

PROVIDER_NAME = "gcp-vertex"


class GCPVertexProvider(
    CompletionProvider[GCPVertexParams],
    EmbeddingProvider[GCPVertexParams],
    PricingProvider,
):
    """GCP Vertex AI provider using the google-genai SDK."""

    def __init__(
        self,
        *,
        auth: AuthProvider["Credentials", "Credentials"] | None = None,
        project: str | None = None,
        location: str | None = None,
        vertexai: bool = True,
        api_key: str | None = None,
    ) -> None:
        self._auth: AuthProvider[Credentials, Credentials] = auth or GCPVertexADCAuthProvider()
        self._project = project
        self._location = location
        self._vertexai = vertexai
        self._api_key = api_key
        self._client: Client | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_gcp_vertex_cost(model, usage)

    # MARK: Client Management

    def _get_client(self) -> "Client":
        if self._client is None:
            credentials = self._auth.get_credentials()
            self._client = create_client(
                vertexai=self._vertexai,
                project=self._project,
                location=self._location,
                credentials=credentials,
                api_key=self._api_key,
            )
        return self._client

    async def _aget_client(self) -> "Client":
        if self._client is None:
            credentials = await self._auth.aget_credentials()
            self._client = create_client(
                vertexai=self._vertexai,
                project=self._project,
                location=self._location,
                credentials=credentials,
                api_key=self._api_key,
            )
        return self._client

    # MARK: Chat

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
        provider_params: GCPVertexParams | None = None,
    ) -> ChatResponse:
        client = self._get_client()
        system, contents = map_messages(messages)
        config = self._build_config(
            system, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_generate_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GCPVertexParams | None = None,
    ) -> ChatResponse:
        client = await self._aget_client()
        system, contents = map_messages(messages)
        config = self._build_config(
            system, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_generate_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GCPVertexParams | None = None,
    ) -> Iterator[ChatChunk]:
        client = self._get_client()
        system, contents = map_messages(messages)
        config = self._build_config(
            system, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
        except Exception as e:
            raise map_gcp_vertex_error(e) from e

        try:
            for chunk in stream:
                mapped = map_generate_content_chunk(chunk, model)
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_gcp_vertex_error(e) from e

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
        provider_params: GCPVertexParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        client = await self._aget_client()
        system, contents = map_messages(messages)
        config = self._build_config(
            system, temperature, max_tokens, top_p, stop, tools, response_format, provider_params
        )
        try:
            stream = client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
            async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]
                mapped = map_generate_content_chunk(chunk, model)  # pyright: ignore[reportUnknownArgumentType]
                if mapped.usage is not None:
                    mapped = mapped.model_copy(update={"cost": self._calculate_cost(model, mapped.usage)})
                yield mapped
        except Exception as e:
            raise map_gcp_vertex_error(e) from e

    # MARK: Embeddings

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: GCPVertexParams | None = None,
    ) -> EmbeddingResponse:
        client = self._get_client()
        contents = input if isinstance(input, list) else [input]
        try:
            response = client.models.embed_content(model=model, contents=contents)
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_embed_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        *,
        provider_params: GCPVertexParams | None = None,
    ) -> EmbeddingResponse:
        client = await self._aget_client()
        contents = input if isinstance(input, list) else [input]
        try:
            response = await client.aio.models.embed_content(model=model, contents=contents)
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_embed_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    # MARK: Internal Helpers

    @staticmethod
    def _build_config(  # noqa: PLR0913
        system_instruction: str | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        response_format: ResponseFormat | None,
        provider_params: GCPVertexParams | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if system_instruction is not None:
            config["system_instruction"] = system_instruction
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if top_p is not None:
            config["top_p"] = top_p
        if stop is not None:
            config["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if tools is not None:
            config["tools"] = map_tools(tools)
        if response_format is not None:
            mime_type, schema = map_response_format(response_format)
            if mime_type is not None:
                config["response_mime_type"] = mime_type
            if schema is not None:
                config["response_schema"] = schema
        if provider_params is not None:
            config.update(GCPVertexProvider._provider_params_kwargs(provider_params))
        return config

    @staticmethod
    def _provider_params_kwargs(params: GCPVertexParams) -> dict[str, Any]:
        """Convert GCPVertexParams to kwargs for GenerateContentConfig."""
        kwargs: dict[str, Any] = {}
        if params.safety_settings is not None:
            kwargs["safety_settings"] = [
                {"category": s.category, "threshold": s.threshold} for s in params.safety_settings
            ]
        if params.presence_penalty is not None:
            kwargs["presence_penalty"] = params.presence_penalty
        if params.frequency_penalty is not None:
            kwargs["frequency_penalty"] = params.frequency_penalty
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.labels is not None:
            kwargs["labels"] = params.labels
        if params.thinking_config is not None:
            kwargs["thinking_config"] = params.thinking_config
        return kwargs
