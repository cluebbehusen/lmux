"""Pydantic wire models for Groq (OpenAI-compatible) response bodies.

Only the fields lmux consumes are declared; unknown fields are ignored (Pydantic's default),
so new server fields do not break parsing.
"""

from pydantic import BaseModel, Field

# MARK: Chat completion (non-streamed)


class WireFunctionCall(BaseModel):
    name: str
    arguments: str


class WireToolCall(BaseModel):
    id: str
    type: str | None = None
    function: WireFunctionCall


class WireMessage(BaseModel):
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[WireToolCall] | None = None


class WireChoice(BaseModel):
    message: WireMessage
    finish_reason: str | None = None


class WirePromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class WireCompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None


class WireUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: WirePromptTokensDetails | None = None
    completion_tokens_details: WireCompletionTokensDetails | None = None


class WireCompletion(BaseModel):
    model: str
    choices: list[WireChoice]
    usage: WireUsage | None = None


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
    reasoning: str | None = None
    tool_calls: list[WireToolCallDelta] | None = None


class WireChunkChoice(BaseModel):
    delta: WireDelta = Field(default_factory=WireDelta)
    finish_reason: str | None = None


class WireChunk(BaseModel):
    model: str | None = None
    choices: list[WireChunkChoice] | None = None
    usage: WireUsage | None = None
