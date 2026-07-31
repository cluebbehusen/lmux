"""Pydantic wire models for the Bedrock Converse (+ ConverseStream) and Titan embedding responses.

Only the fields lmux consumes are declared; unknown fields are retained so provider continuations
can replay them without core changes. Bedrock uses camelCase on the wire, so a shared alias
generator maps snake_case field names to camelCase (``tool_use_id`` -> ``toolUseId``).
Content blocks and stream events are field-bags (no ``type`` tag), so each carries the shapes
lmux reads as optional fields and the mapper dispatches on which are present. Nested containers
default to empty (mirroring the old ``dict.get(..., {})`` access) so the mappers stay guard-free.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _BedrockModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")


# MARK: Token usage (shared by the response and the stream metadata event)


class WireCacheDetail(_BedrockModel):
    ttl: str
    input_tokens: int


class WireTokenUsage(_BedrockModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cache_details: list[WireCacheDetail] | None = None


# MARK: Converse response


class WireToolUse(_BedrockModel):
    tool_use_id: str
    name: str
    input: dict[str, Any]


class WireReasoningText(_BedrockModel):
    text: str | None = None
    signature: str | None = None


class WireReasoningContent(_BedrockModel):
    reasoning_text: WireReasoningText | None = None
    redacted_content: str | None = None


class WireContentBlock(_BedrockModel):
    text: str | None = None
    tool_use: WireToolUse | None = None
    reasoning_content: WireReasoningContent | None = None


class WireMessage(_BedrockModel):
    content: list[WireContentBlock] = Field(default_factory=list)


class WireOutput(_BedrockModel):
    message: WireMessage = Field(default_factory=WireMessage)


class WireConverseResponse(_BedrockModel):
    output: WireOutput = Field(default_factory=WireOutput)
    stop_reason: str | None = None
    usage: WireTokenUsage | None = None


# MARK: ConverseStream events


class WireToolUseDelta(_BedrockModel):
    input: str = ""


class WireReasoningDelta(_BedrockModel):
    text: str | None = None
    signature: str | None = None
    redacted_content: str | None = None


class WireStreamDelta(_BedrockModel):
    text: str | None = None
    tool_use: WireToolUseDelta | None = None
    reasoning_content: WireReasoningDelta | None = None


class WireContentBlockDeltaEvent(_BedrockModel):
    delta: WireStreamDelta = Field(default_factory=WireStreamDelta)
    content_block_index: int = 0


class WireToolUseStart(_BedrockModel):
    tool_use_id: str | None = None
    name: str | None = None


class WireContentBlockStart(_BedrockModel):
    tool_use: WireToolUseStart | None = None


class WireContentBlockStartEvent(_BedrockModel):
    start: WireContentBlockStart = Field(default_factory=WireContentBlockStart)
    content_block_index: int = 0


class WireMessageStopEvent(_BedrockModel):
    stop_reason: str | None = None


class WireMetadataEvent(_BedrockModel):
    usage: WireTokenUsage | None = None


class WireStreamEvent(_BedrockModel):
    content_block_delta: WireContentBlockDeltaEvent | None = None
    content_block_start: WireContentBlockStartEvent | None = None
    message_stop: WireMessageStopEvent | None = None
    metadata: WireMetadataEvent | None = None


# MARK: Embeddings (Titan InvokeModel)


class WireEmbeddingResponse(_BedrockModel):
    embedding: list[float] = Field(default_factory=list)
    input_text_token_count: int = 0
