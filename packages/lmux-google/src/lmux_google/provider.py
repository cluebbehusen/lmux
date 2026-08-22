"""Google (Gemini) provider implementation (SDK-lite, httpx transport)."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Literal, override

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

from lmux._http import aiter_sse, iter_sse
from lmux.cost import ModelPricing, calculate_cost
from lmux.exceptions import LmuxError, ProviderError
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
from lmux_google._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_body,
    raise_for_status,
)
from lmux_google._lazy import (
    GEMINI_BASE_URL,
    api_key_headers,
    bearer_headers,
    bearer_token,
    create_async_client,
    create_sync_client,
    merge_headers,
    vertex_base_url,
)
from lmux_google._mappers import (
    GoogleContinuationState,
    Json,
    map_batch_embeddings_response,
    map_embed_content_response,
    map_generate_content_chunk,
    map_generate_content_response,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
    map_vertex_embed_response,
)
from lmux_google._wire import (
    WireBatchEmbeddingsResponse,
    WireEmbedContentResponse,
    WireGenerateContentResponse,
    WireVertexPredictResponse,
)
from lmux_google.auth import GoogleADCAuthProvider
from lmux_google.cost import (
    VERTEX_NON_GLOBAL_MULTIPLIER,
    VERTEX_NON_GLOBAL_PREMIUM_START,
    apply_cost_multiplier,
    calculate_google_cost,
    has_vertex_non_global_premium,
)
from lmux_google.params import GoogleParams, GoogleSearchConfig

type GoogleAuth = AuthProvider["Credentials | str", "Credentials | str"]

PROVIDER_NAME = "google"

_HTTP_ERROR = 400
_THINKING_BUDGETS: dict[str, int] = {"low": 1024, "medium": 8192, "high": 32768}


def _today() -> date:
    """Return today's date, indirected so tests can pin the pricing clock."""
    return date.today()


class GoogleProvider(
    CompletionProvider[GoogleParams],
    EmbeddingProvider[GoogleParams],
    PricingProvider,
):
    """Google provider over httpx (Vertex AI or the Gemini Developer API)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        auth: GoogleAuth | None = None,
        project: str | None = None,
        location: str | None = None,
        vertexai: bool = True,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        transport: "httpx.BaseTransport | None" = None,
        async_transport: "httpx.AsyncBaseTransport | None" = None,
    ) -> None:
        self._auth: GoogleAuth = auth or GoogleADCAuthProvider()
        self._project: str | None = project
        self._location: str | None = location
        self._vertexai: bool = vertexai
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._default_headers: Mapping[str, str] | None = default_headers
        self._transport: httpx.BaseTransport | None = transport
        self._async_transport: httpx.AsyncBaseTransport | None = async_transport
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._credentials: Credentials | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    @staticmethod
    def _resolve_pricing_as_of(provider_params: GoogleParams | None) -> date:
        """Effective pricing date: an explicit ``pricing_as_of`` override, else today."""
        if provider_params is not None and provider_params.pricing_as_of is not None:
            return provider_params.pricing_as_of
        return _today()

    def _vertex_multiplier(self, model: str, as_of: date | None) -> float:
        """Non-global Vertex endpoints bill a 10% premium on GA Gemini 3+ models.

        The global endpoint (the default when no location is set) bills list
        prices, and the Gemini Developer API has no endpoint premium at all.
        The premium only exists from ``VERTEX_NON_GLOBAL_PREMIUM_START``, so a
        cost replayed against an earlier date takes no multiplier; ``as_of`` of
        ``None`` means current pricing.
        """
        if not self._vertexai or (self._location or "global") == "global":
            return 1.0
        if as_of is not None and as_of < VERTEX_NON_GLOBAL_PREMIUM_START:
            return 1.0
        if has_vertex_non_global_premium(model):
            return VERTEX_NON_GLOBAL_MULTIPLIER
        return 1.0

    def _calculate_cost(self, model: str, usage: Usage, as_of: date | None = None) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            cost = calculate_cost(usage, pricing, as_of)
        else:
            cost = calculate_google_cost(model, usage, as_of)
        if cost is None:
            return None
        multiplier = self._vertex_multiplier(model, as_of)
        if multiplier == 1.0:
            return cost
        return apply_cost_multiplier(cost, multiplier)

    def _cost_fn_for(self, as_of: date) -> "Callable[[str, Usage], Cost | None]":
        """A cost callable pinned to a pricing date, for the response mappers."""
        return lambda model, usage: self._calculate_cost(model, usage, as_of)

    def _cost_fn(self, provider_params: GoogleParams | None) -> "Callable[[str, Usage], Cost | None]":
        """A cost callable pinned to this request's pricing date, for the response mappers."""
        return self._cost_fn_for(self._resolve_pricing_as_of(provider_params))

    # MARK: Client Management

    def _base_url(self) -> str:
        return vertex_base_url(self._location) if self._vertexai else GEMINI_BASE_URL

    def _static_headers(self, auth_result: "Credentials | str") -> Mapping[str, str]:
        # API keys are static and baked into the cached client; Vertex bearer tokens are
        # short-lived and applied per request (see _request_headers) so they refresh rather
        # than freeze on a long-lived provider.
        if isinstance(auth_result, str):
            return api_key_headers(auth_result, self._default_headers)
        return merge_headers(self._default_headers, {"Content-Type": "application/json"})

    def _request_headers(self) -> Mapping[str, str]:
        if self._credentials is None:
            return {}
        return bearer_headers(bearer_token(self._credentials), self._credentials.quota_project_id)

    async def _arequest_headers(self) -> Mapping[str, str]:
        if self._credentials is None:
            return {}
        # Credential refresh does blocking HTTP; run it off the event loop.
        token = await asyncio.to_thread(bearer_token, self._credentials)
        return bearer_headers(token, self._credentials.quota_project_id)

    def _path(self, model: str, method: str, *, sse: bool = False) -> str:
        suffix = "?alt=sse" if sse else ""
        if self._vertexai:
            if self._project is None:
                msg = "Vertex AI requires a project; pass project=... to GoogleProvider or use vertexai=False"
                raise ProviderError(msg, provider=PROVIDER_NAME)
            location = self._location or "global"
            base = f"/v1/projects/{self._project}/locations/{location}/publishers/google/models"
            return f"{base}/{model}:{method}{suffix}"
        return f"/v1beta/models/{model}:{method}{suffix}"

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            auth_result = self._auth.get_credentials()
            self._credentials = auth_result if not isinstance(auth_result, str) else None
            self._sync_client = create_sync_client(
                base_url=self._base_url(),
                headers=self._static_headers(auth_result),
                timeout=self._timeout,
                max_retries=self._max_retries,
                transport=self._transport,
            )
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            auth_result = await self._auth.aget_credentials()
            self._credentials = auth_result if not isinstance(auth_result, str) else None
            self._async_client = create_async_client(
                base_url=self._base_url(),
                headers=self._static_headers(auth_result),
                timeout=self._timeout,
                max_retries=self._max_retries,
                transport=self._async_transport,
            )
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
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
        provider_params: GoogleParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
            vertexai=self._vertexai,
        )  # fmt: skip
        path = self._path(model, "generateContent")
        try:
            client = self._get_sync_client()
            response = client.post(path, json=body, headers=self._request_headers())
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_generate_content_response(
            parse_body(response, WireGenerateContentResponse),
            model,
            PROVIDER_NAME,
            self._cost_fn(provider_params),
            vertexai=self._vertexai,
        )

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
        provider_params: GoogleParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
            vertexai=self._vertexai,
        )  # fmt: skip
        path = self._path(model, "generateContent")
        try:
            client = await self._get_async_client()
            response = await client.post(path, json=body, headers=await self._arequest_headers())
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_generate_content_response(
            parse_body(response, WireGenerateContentResponse),
            model,
            PROVIDER_NAME,
            self._cost_fn(provider_params),
            vertexai=self._vertexai,
        )

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
        provider_params: GoogleParams | None = None,
    ) -> Iterator[ChatChunk]:
        body = self._build_body(
            messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
            vertexai=self._vertexai,
        )  # fmt: skip
        path = self._path(model, "streamGenerateContent", sse=True)
        try:
            client = self._get_sync_client()
            headers = self._request_headers()
        except Exception as e:
            raise map_transport_error(e) from e
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            continuation_state = GoogleContinuationState(vertexai=self._vertexai)
            with client.stream("POST", path, json=body, headers=headers) as response:
                if response.status_code >= _HTTP_ERROR:
                    response.read()
                    raise error_from_response(response)  # noqa: TRY301
                for _event, data in iter_sse(response):
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    wire = WireGenerateContentResponse.model_validate(chunk)
                    part_offset = continuation_state.add(wire)
                    mapped = self._map_stream_chunk(wire, model, as_of, part_offset=part_offset)
                    if mapped.finish_reason is not None:
                        mapped = mapped.model_copy(update={"continuation": continuation_state.continuation()})
                    yield mapped
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

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
        provider_params: GoogleParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        body = self._build_body(
            messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
            vertexai=self._vertexai,
        )  # fmt: skip
        path = self._path(model, "streamGenerateContent", sse=True)
        try:
            client = await self._get_async_client()
            headers = await self._arequest_headers()
        except Exception as e:
            raise map_transport_error(e) from e
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            continuation_state = GoogleContinuationState(vertexai=self._vertexai)
            async with client.stream("POST", path, json=body, headers=headers) as response:
                if response.status_code >= _HTTP_ERROR:
                    await response.aread()
                    raise error_from_response(response)  # noqa: TRY301
                async for _event, data in aiter_sse(response):
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    wire = WireGenerateContentResponse.model_validate(chunk)
                    part_offset = continuation_state.add(wire)
                    mapped = self._map_stream_chunk(wire, model, as_of, part_offset=part_offset)
                    if mapped.finish_reason is not None:
                        mapped = mapped.model_copy(update={"continuation": continuation_state.continuation()})
                    yield mapped
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

    # MARK: Embeddings

    @override
    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: GoogleParams | None = None,
    ) -> EmbeddingResponse:
        as_of = self._resolve_pricing_as_of(provider_params)
        if self._vertexai and _uses_embed_content(model):
            return self._embed_content(model, input, dimensions, provider_params, as_of)
        texts = input if isinstance(input, list) else [input]
        if self._vertexai and model.startswith("gemini-embedding-001") and not texts:
            return self._embedding_response(model, [], 0, as_of)
        try:
            client = self._get_sync_client()
            path, body = self._embed_path_and_body(model, texts, dimensions, provider_params)
            response = client.post(path, json=body, headers=self._request_headers())
            raise_for_status(response)
            result = self._map_embed_response(response, model, as_of)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e
        return result

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: GoogleParams | None = None,
    ) -> EmbeddingResponse:
        as_of = self._resolve_pricing_as_of(provider_params)
        if self._vertexai and _uses_embed_content(model):
            return await self._aembed_content(model, input, dimensions, provider_params, as_of)
        texts = input if isinstance(input, list) else [input]
        if self._vertexai and model.startswith("gemini-embedding-001") and not texts:
            return self._embedding_response(model, [], 0, as_of)
        try:
            client = await self._get_async_client()
            path, body = self._embed_path_and_body(model, texts, dimensions, provider_params)
            response = await client.post(path, json=body, headers=await self._arequest_headers())
            raise_for_status(response)
            result = self._map_embed_response(response, model, as_of)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e
        return result

    def _embed_content(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: GoogleParams | None,
        as_of: date,
    ) -> EmbeddingResponse:
        # Vertex serves Gemini Embedding 2 models only through :embedContent, which takes a single
        # content per request (no batch), so a list is issued one item at a time.
        texts = input if isinstance(input, list) else [input]
        path = self._path(model, "embedContent")
        embeddings: list[list[float]] = []
        input_tokens = 0
        try:
            client = self._get_sync_client()
            for text in texts:
                headers = self._request_headers()
                response = client.post(
                    path, json=_embed_content_body(text, dimensions, provider_params), headers=headers
                )
                raise_for_status(response)
                values, tokens = map_embed_content_response(parse_body(response, WireEmbedContentResponse))
                embeddings.append(values)
                input_tokens += tokens
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e
        return self._embedding_response(model, embeddings, input_tokens, as_of)

    async def _aembed_content(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: GoogleParams | None,
        as_of: date,
    ) -> EmbeddingResponse:
        texts = input if isinstance(input, list) else [input]
        path = self._path(model, "embedContent")
        embeddings: list[list[float]] = []
        input_tokens = 0
        try:
            client = await self._get_async_client()
            for text in texts:
                headers = await self._arequest_headers()
                response = await client.post(
                    path, json=_embed_content_body(text, dimensions, provider_params), headers=headers
                )
                raise_for_status(response)
                values, tokens = map_embed_content_response(parse_body(response, WireEmbedContentResponse))
                embeddings.append(values)
                input_tokens += tokens
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e
        return self._embedding_response(model, embeddings, input_tokens, as_of)

    def _embedding_response(
        self, model: str, embeddings: list[list[float]], input_tokens: int, as_of: date
    ) -> EmbeddingResponse:
        usage = Usage(input_tokens=input_tokens, output_tokens=0)
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=usage,
            cost=self._calculate_cost(model, usage, as_of),
            model=model,
            provider=PROVIDER_NAME,
        )

    # MARK: Internal Helpers

    def _map_stream_chunk(
        self, chunk: WireGenerateContentResponse, model: str, as_of: date, *, part_offset: int = 0
    ) -> ChatChunk:
        mapped = map_generate_content_chunk(chunk, model, PROVIDER_NAME, part_offset=part_offset)
        if mapped.usage is not None:
            mapped = mapped.model_copy(update={"cost": self._calculate_cost(model, mapped.usage, as_of)})
        return mapped

    def _embed_path_and_body(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: GoogleParams | None,
    ) -> tuple[str, Json]:
        # Vertex AI serves embeddings through the generic ``:predict`` endpoint (instances/parameters),
        # while the Gemini Developer API uses ``:batchEmbedContents`` (requests/content).
        if self._vertexai:
            return self._path(model, "predict"), self._build_vertex_embed_body(input, dimensions, provider_params)
        body = self._build_embed_body(model, input, dimensions, provider_params)
        return self._path(model, "batchEmbedContents"), body

    def _map_embed_response(self, response: "httpx.Response", model: str, as_of: date) -> EmbeddingResponse:
        cost_fn = self._cost_fn_for(as_of)
        if self._vertexai:
            return map_vertex_embed_response(
                parse_body(response, WireVertexPredictResponse), model, PROVIDER_NAME, cost_fn
            )
        return map_batch_embeddings_response(
            parse_body(response, WireBatchEmbeddingsResponse), model, PROVIDER_NAME, cost_fn
        )

    @staticmethod
    def _build_embed_body(
        model: str,
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: GoogleParams | None,
    ) -> Json:
        texts = input if isinstance(input, list) else [input]
        requests: list[Json] = []
        for text in texts:
            req: Json = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
            if dimensions is not None:
                req["outputDimensionality"] = dimensions
            if provider_params is not None and provider_params.task_type is not None:
                req["taskType"] = provider_params.task_type
            requests.append(req)
        return {"requests": requests}

    @staticmethod
    def _build_vertex_embed_body(
        input: str | list[str],  # noqa: A002
        dimensions: int | None,
        provider_params: GoogleParams | None,
    ) -> Json:
        texts = input if isinstance(input, list) else [input]
        instances: list[Json] = []
        for text in texts:
            instance: Json = {"content": text}
            if provider_params is not None and provider_params.task_type is not None:
                instance["task_type"] = provider_params.task_type
            instances.append(instance)
        body: Json = {"instances": instances}
        if dimensions is not None:
            body["parameters"] = {"outputDimensionality": dimensions}
        return body

    @staticmethod
    def _build_body(  # noqa: PLR0913, PLR0912
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        provider_params: GoogleParams | None,
        *,
        vertexai: bool,
    ) -> Json:
        system, contents = map_messages(messages, include_tool_call_ids=not vertexai)
        body: Json = {"contents": contents}
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen: Json = {}
        if temperature is not None:
            gen["temperature"] = temperature
        if max_tokens is not None:
            gen["maxOutputTokens"] = max_tokens
        if top_p is not None:
            gen["topP"] = top_p
        if stop is not None:
            gen["stopSequences"] = [stop] if isinstance(stop, str) else stop
        if response_format is not None:
            mime_type, schema = map_response_format(response_format)
            if mime_type is not None:
                gen["responseMimeType"] = mime_type
            if schema is not None:
                # responseJsonSchema accepts full JSON Schema (incl. $defs/$ref); responseSchema
                # is the narrower OpenAPI-style shape that rejects those constructs.
                gen["responseJsonSchema"] = schema
        if reasoning_effort is not None:
            gen["thinkingConfig"] = {"thinkingBudget": _THINKING_BUDGETS[reasoning_effort], "includeThoughts": True}
        if tools is not None:
            body["tools"] = map_tools(tools)
        if tool_choice is not None:
            body["toolConfig"] = map_tool_choice(tool_choice)
        if provider_params is not None:
            GoogleProvider._apply_provider_params(body, gen, provider_params)
        if gen:
            body["generationConfig"] = gen
        return body

    @staticmethod
    def _apply_provider_params(body: Json, gen: Json, params: GoogleParams) -> None:
        """Merge GoogleParams into the request ``body`` and ``generationConfig`` in place."""
        if params.safety_settings is not None:
            body["safetySettings"] = [
                {"category": s.category, "threshold": s.threshold} for s in params.safety_settings
            ]
        if params.presence_penalty is not None:
            gen["presencePenalty"] = params.presence_penalty
        if params.frequency_penalty is not None:
            gen["frequencyPenalty"] = params.frequency_penalty
        if params.seed is not None:
            gen["seed"] = params.seed
        if params.labels is not None:
            body["labels"] = params.labels
        if params.thinking_config is not None:
            gen["thinkingConfig"] = params.thinking_config
        special_tools = GoogleProvider._build_special_tools(params)
        if special_tools:
            body.setdefault("tools", []).extend(special_tools)

    @staticmethod
    def _build_special_tools(params: GoogleParams) -> list[Json]:
        """Convert special tool params to Gemini tool dicts."""
        tools: list[Json] = []
        if params.google_search is not None:
            if params.google_search is True:
                tools.append({"googleSearch": {}})
            elif isinstance(params.google_search, GoogleSearchConfig):
                tools.append({"googleSearch": GoogleProvider._build_google_search_dict(params.google_search)})
        if params.google_search_retrieval is not None:
            gsr_dict: Json = {}
            drc = params.google_search_retrieval.dynamic_retrieval_config
            if drc is not None:
                drc_dict: Json = {}
                if drc.mode is not None:
                    drc_dict["mode"] = drc.mode
                if drc.dynamic_threshold is not None:
                    drc_dict["dynamicThreshold"] = drc.dynamic_threshold
                if drc_dict:
                    gsr_dict["dynamicRetrievalConfig"] = drc_dict
            tools.append({"googleSearchRetrieval": gsr_dict})
        if params.code_execution is True:
            tools.append({"codeExecution": {}})
        return tools

    @staticmethod
    def _build_google_search_dict(config: GoogleSearchConfig) -> Json:
        """Convert GoogleSearchConfig to a Gemini tool dict."""
        gs_dict: Json = {}
        if config.search_types is not None:
            st_dict: Json = {}
            if config.search_types.web_search is True:
                st_dict["webSearch"] = {}
            if config.search_types.image_search is True:
                st_dict["imageSearch"] = {}
            if st_dict:
                gs_dict["searchTypes"] = st_dict
        if config.exclude_domains is not None:
            gs_dict["excludeDomains"] = config.exclude_domains
        return gs_dict


def _uses_embed_content(model: str) -> bool:
    """Gemini Embedding 2 models are served on Vertex only through :embedContent (not :predict)."""
    return model.startswith("gemini-embedding-2")


def _embed_content_body(text: str, dimensions: int | None, provider_params: GoogleParams | None) -> Json:
    body: Json = {"content": {"parts": [{"text": text}]}}
    if dimensions is not None:
        body["outputDimensionality"] = dimensions
    if provider_params is not None and provider_params.task_type is not None:
        body["embedContentConfig"] = {"taskType": provider_params.task_type}
    return body
