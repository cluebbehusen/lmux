"""Pydantic wire models for Azure AI Foundry (OpenAI-compatible) response bodies.

Only the fields lmux consumes are declared; unknown fields are ignored (Pydantic's default),
so new server fields do not break parsing. Covers the three response families the provider
reads: chat completions (and streaming chunks), embeddings, and the Responses API.
"""

from pydantic import BaseModel, Field

# MARK: Chat completion (non-streamed)


class WireFunctionCall(BaseModel):
    name: str
    arguments: str


class WireToolCall(BaseModel):
    id: str
    type: str | None = None
    # Absent for non-function tool calls; lmux keeps only function-typed calls.
    function: WireFunctionCall | None = None


class WireMessage(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[WireToolCall] | None = None


class WireChoice(BaseModel):
    message: WireMessage
    finish_reason: str | None = None


class WirePromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class WireCompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None


class WireCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: WirePromptTokensDetails | None = None
    completion_tokens_details: WireCompletionTokensDetails | None = None


class WireCompletion(BaseModel):
    model: str
    choices: list[WireChoice]
    usage: WireCompletionUsage | None = None


# MARK: Streaming chunk


class WireFunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class WireToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: str | None = None
    function: WireFunctionDelta | None = None


class WireDelta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[WireToolCallDelta] | None = None


class WireChunkChoice(BaseModel):
    delta: WireDelta = Field(default_factory=WireDelta)
    finish_reason: str | None = None


class WireChunk(BaseModel):
    model: str | None = None
    choices: list[WireChunkChoice] | None = None
    usage: WireCompletionUsage | None = None


# MARK: Embeddings


class WireEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class WireEmbeddingUsage(BaseModel):
    prompt_tokens: int


class WireEmbeddingResponse(BaseModel):
    model: str
    data: list[WireEmbeddingItem]
    usage: WireEmbeddingUsage


# MARK: Responses API


class WireOutputContent(BaseModel):
    type: str
    text: str | None = None


class WireOutputItem(BaseModel):
    type: str
    content: list[WireOutputContent] | None = None


class WireResponsesInputDetails(BaseModel):
    cached_tokens: int | None = None


class WireResponsesOutputDetails(BaseModel):
    reasoning_tokens: int | None = None


class WireResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    input_tokens_details: WireResponsesInputDetails | None = None
    output_tokens_details: WireResponsesOutputDetails | None = None


class WireResponsesResponse(BaseModel):
    id: str
    model: str
    output: list[WireOutputItem] | None = None
    usage: WireResponsesUsage | None = None
