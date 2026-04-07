"""GCP Vertex AI provider implementation."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, override

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
    ToolChoice,
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
    map_tool_choice,
    map_tools,
)
from lmux_gcp_vertex.auth import GCPVertexADCAuthProvider
from lmux_gcp_vertex.cost import calculate_gcp_vertex_cost
from lmux_gcp_vertex.params import GCPVertexParams, GoogleSearchConfig

type GCPVertexAuth = AuthProvider["Credentials | str", "Credentials | str"]

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
        auth: GCPVertexAuth | None = None,
        project: str | None = None,
        location: str | None = None,
        vertexai: bool = True,
    ) -> None:
        self._auth: GCPVertexAuth = auth or GCPVertexADCAuthProvider()
        self._project: str | None = project
        self._location: str | None = location
        self._vertexai: bool = vertexai
        self._client: Client | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
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
            auth_result = self._auth.get_credentials()
            credentials, api_key = (None, auth_result) if isinstance(auth_result, str) else (auth_result, None)
            self._client = create_client(
                vertexai=self._vertexai,
                project=self._project,
                location=self._location,
                credentials=credentials,
                api_key=api_key,
            )
        return self._client

    async def _aget_client(self) -> "Client":
        loop = asyncio.get_running_loop()
        if self._client is None or self._async_loop is not loop:
            self._client = None
            auth_result = await self._auth.aget_credentials()
            credentials, api_key = (None, auth_result) if isinstance(auth_result, str) else (auth_result, None)
            self._client = create_client(
                vertexai=self._vertexai,
                project=self._project,
                location=self._location,
                credentials=credentials,
                api_key=api_key,
            )
            self._async_loop = loop
        return self._client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._client is not None:
            await self._client.aio.aclose()
            self._client = None
            self._async_loop = None

    # MARK: Chat

    @override
    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> ChatResponse:
        system, contents = map_messages(messages)
        config = self._build_config(
            system,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_generate_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    @override
    async def achat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> ChatResponse:
        system, contents = map_messages(messages)
        config = self._build_config(
            system,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = await self._aget_client()
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,  # pyright: ignore[reportArgumentType]
            )
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_generate_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    @override
    def chat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> Iterator[ChatChunk]:
        system, contents = map_messages(messages)
        config = self._build_config(
            system,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_client()
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

    @override
    async def achat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        system, contents = map_messages(messages)
        config = self._build_config(
            system,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = await self._aget_client()
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

    @override
    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> EmbeddingResponse:
        contents = input if isinstance(input, list) else [input]
        config = self._build_embed_config(dimensions, provider_params)
        try:
            client = self._get_client()
            response = client.models.embed_content(model=model, contents=contents, config=config)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_embed_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: GCPVertexParams | None = None,
    ) -> EmbeddingResponse:
        contents = input if isinstance(input, list) else [input]
        config = self._build_embed_config(dimensions, provider_params)
        try:
            client = await self._aget_client()
            response = await client.aio.models.embed_content(model=model, contents=contents, config=config)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise map_gcp_vertex_error(e) from e
        return map_embed_content_response(response, model, PROVIDER_NAME, self._calculate_cost)

    # MARK: Internal Helpers

    @staticmethod
    def _build_embed_config(
        dimensions: int | None,
        provider_params: GCPVertexParams | None,
    ) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        if dimensions is not None:
            config["output_dimensionality"] = dimensions
        if provider_params is not None and provider_params.task_type is not None:
            config["task_type"] = provider_params.task_type
        return config or None

    @staticmethod
    def _build_config(  # noqa: PLR0913, PLR0912
        system_instruction: str | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
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
        if tool_choice is not None:
            config["tool_config"] = map_tool_choice(tool_choice)
        if response_format is not None:
            mime_type, schema = map_response_format(response_format)
            if mime_type is not None:
                config["response_mime_type"] = mime_type
            if schema is not None:
                config["response_schema"] = schema
        if reasoning_effort is not None:
            budget = {"low": 1024, "medium": 8192, "high": 32768}[reasoning_effort]
            config["thinking_config"] = {"thinking_budget": budget, "include_thoughts": True}
        if provider_params is not None:
            config.update(GCPVertexProvider._provider_params_kwargs(provider_params))
            special_tools = GCPVertexProvider._build_special_tools(provider_params)
            if special_tools:
                config.setdefault("tools", []).extend(special_tools)
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

    @staticmethod
    def _build_special_tools(params: GCPVertexParams) -> list[dict[str, Any]]:
        """Convert special tool params to google-genai tool dicts."""
        tools: list[dict[str, Any]] = []
        if params.google_search is not None:
            if params.google_search is True:
                tools.append({"google_search": {}})
            elif isinstance(params.google_search, GoogleSearchConfig):
                tools.append({"google_search": GCPVertexProvider._build_google_search_dict(params.google_search)})
        if params.google_search_retrieval is not None:
            gsr_dict: dict[str, Any] = {}
            drc = params.google_search_retrieval.dynamic_retrieval_config
            if drc is not None:
                drc_dict: dict[str, Any] = {}
                if drc.mode is not None:
                    drc_dict["mode"] = drc.mode
                if drc.dynamic_threshold is not None:
                    drc_dict["dynamic_threshold"] = drc.dynamic_threshold
                if drc_dict:
                    gsr_dict["dynamic_retrieval_config"] = drc_dict
            tools.append({"google_search_retrieval": gsr_dict})
        if params.code_execution is True:
            tools.append({"code_execution": {}})
        return tools

    @staticmethod
    def _build_google_search_dict(config: GoogleSearchConfig) -> dict[str, Any]:
        """Convert GoogleSearchConfig to a google-genai tool dict."""
        gs_dict: dict[str, Any] = {}
        if config.search_types is not None:
            st_dict: dict[str, Any] = {}
            if config.search_types.web_search is True:
                st_dict["web_search"] = {}
            if config.search_types.image_search is True:
                st_dict["image_search"] = {}
            if st_dict:
                gs_dict["search_types"] = st_dict
        if config.exclude_domains is not None:
            gs_dict["exclude_domains"] = config.exclude_domains
        return gs_dict
