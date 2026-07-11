"""Pydantic wire models for the Anthropic Messages API response and stream events.

Only the fields lmux consumes are declared; unknown fields are ignored (Pydantic's default), so
new server fields do not break parsing. Content blocks and streaming deltas are discriminated
unions with an explicit unknown-tag fallback, so an unrecognized ``type`` validates into the
fallback variant instead of raising.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Tag

# MARK: Usage (shared by the response, message_start, and message_delta)


class WireCacheCreation(BaseModel):
    ephemeral_5m_input_tokens: int | None = None
    ephemeral_1h_input_tokens: int | None = None


class WireUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_creation: WireCacheCreation | None = None


class WireDeltaUsage(BaseModel):
    """The message_delta ``usage`` object carries only output tokens; input usage came in message_start."""

    output_tokens: int


# MARK: Content blocks (response ``content[]`` and content_block_start)

_KNOWN_BLOCKS = frozenset({"text", "thinking", "tool_use"})


class WireTextBlock(BaseModel):
    type: Literal["text"]
    text: str


class WireThinkingBlock(BaseModel):
    type: Literal["thinking"]
    thinking: str


class WireToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class WireUnknownBlock(BaseModel):
    """Fallback for block types lmux does not consume (e.g. ``redacted_thinking``)."""

    type: str


def _content_block_tag(value: Any) -> str:  # noqa: ANN401 — pydantic passes the raw input
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return tag if tag in _KNOWN_BLOCKS else "unknown"


WireContentBlock = Annotated[
    Annotated[WireTextBlock, Tag("text")]
    | Annotated[WireThinkingBlock, Tag("thinking")]
    | Annotated[WireToolUseBlock, Tag("tool_use")]
    | Annotated[WireUnknownBlock, Tag("unknown")],
    Discriminator(_content_block_tag),
]


# MARK: Streaming content_block_delta payloads

_KNOWN_DELTAS = frozenset({"text_delta", "input_json_delta", "thinking_delta"})


class WireTextDelta(BaseModel):
    type: Literal["text_delta"]
    text: str


class WireInputJsonDelta(BaseModel):
    type: Literal["input_json_delta"]
    partial_json: str


class WireThinkingDelta(BaseModel):
    type: Literal["thinking_delta"]
    thinking: str


class WireUnknownDelta(BaseModel):
    """Fallback for delta types lmux does not consume (e.g. ``signature_delta``)."""

    type: str


def _stream_delta_tag(value: Any) -> str:  # noqa: ANN401 — pydantic passes the raw input
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return tag if tag in _KNOWN_DELTAS else "unknown"


WireStreamDelta = Annotated[
    Annotated[WireTextDelta, Tag("text_delta")]
    | Annotated[WireInputJsonDelta, Tag("input_json_delta")]
    | Annotated[WireThinkingDelta, Tag("thinking_delta")]
    | Annotated[WireUnknownDelta, Tag("unknown")],
    Discriminator(_stream_delta_tag),
]


# MARK: Non-streamed response


class WireMessage(BaseModel):
    model: str
    content: list[WireContentBlock]
    usage: WireUsage
    stop_reason: str | None = None


# MARK: Stream events


class WireStartMessage(BaseModel):
    model: str
    usage: WireUsage


class WireMessageStartEvent(BaseModel):
    message: WireStartMessage


class WireContentBlockStartEvent(BaseModel):
    index: int
    content_block: WireContentBlock


class WireContentBlockDeltaEvent(BaseModel):
    index: int
    delta: WireStreamDelta


class WireMessageDeltaBody(BaseModel):
    stop_reason: str | None = None


class WireMessageDeltaEvent(BaseModel):
    delta: WireMessageDeltaBody
    usage: WireDeltaUsage
