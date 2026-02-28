"""Core type definitions for lmux."""

from typing import Literal

from pydantic import BaseModel

# MARK: Provider Params


class BaseProviderParams(BaseModel):
    """Base class for provider-specific parameter types."""


# MARK: Content Parts


class TextContent(BaseModel):
    """Text content in a message."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content in a message, referenced by URL or base64 data URI."""

    type: Literal["image_url"] = "image_url"
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


type ContentPart = TextContent | ImageContent


# MARK: Tools


class FunctionDefinition(BaseModel):
    """JSON Schema-based function definition for tool calling."""

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    strict: bool | None = None


class Tool(BaseModel):
    """A tool the model may call."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCallResult(BaseModel):
    """The function name and arguments in a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A complete tool call from the model."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCallResult


class FunctionCallDelta(BaseModel):
    """Incremental function call data in a streaming chunk."""

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """Incremental tool call data in a streaming chunk."""

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


# MARK: Messages


class SystemMessage(BaseModel):
    """System/instruction message."""

    role: Literal["system"] = "system"
    content: str


class DeveloperMessage(BaseModel):
    """Developer message (for o-series models)."""

    role: Literal["developer"] = "developer"
    content: str


class UserMessage(BaseModel):
    """User message with text or multimodal content."""

    role: Literal["user"] = "user"
    content: str | list[ContentPart]


class AssistantMessage(BaseModel):
    """Assistant message, possibly with tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    """Tool result message."""

    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


type Message = SystemMessage | DeveloperMessage | UserMessage | AssistantMessage | ToolMessage


# MARK: Response Format


class TextResponseFormat(BaseModel):
    """Request plain text output."""

    type: Literal["text"] = "text"


class JsonObjectResponseFormat(BaseModel):
    """Request JSON object output."""

    type: Literal["json_object"] = "json_object"


class JsonSchemaResponseFormat(BaseModel):
    """Request structured output matching a JSON schema."""

    type: Literal["json_schema"] = "json_schema"
    name: str
    json_schema: dict[str, object]
    description: str | None = None
    strict: bool | None = None


type ResponseFormat = TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat


# MARK: Usage & Cost


class Usage(BaseModel):
    """Token usage for a request."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class Cost(BaseModel):
    """Monetary cost breakdown for a request."""

    input_cost: float
    output_cost: float
    total_cost: float
    cache_read_cost: float | None = None
    cache_creation_cost: float | None = None
    currency: str = "USD"


# MARK: Chat Response


class ChatResponse(BaseModel):
    """Flattened chat completion response."""

    content: str | None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None
    cost: Cost | None
    model: str
    provider: str
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    """A single chunk in a streaming chat response."""

    delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] | None = None
    usage: Usage | None = None
    cost: Cost | None = None
    finish_reason: str | None = None
    model: str | None = None


# MARK: Embedding Response


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    embeddings: list[list[float]]
    usage: Usage
    cost: Cost | None
    model: str
    provider: str


# MARK: Responses API Input


class ResponseInputMessage(BaseModel):
    """A message item in Responses API input."""

    role: Literal["user", "assistant", "system", "developer"]
    content: str


class ResponseInputFunctionCall(BaseModel):
    """A function call item in Responses API input (for multi-turn tool use)."""

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class ResponseInputFunctionCallOutput(BaseModel):
    """A function call output item in Responses API input."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


type ResponseInputItem = ResponseInputMessage | ResponseInputFunctionCall | ResponseInputFunctionCallOutput


# MARK: Responses API Response


class ResponseResponse(BaseModel):
    """Response from the Responses API (OpenAI Responses API style)."""

    id: str
    output_text: str
    usage: Usage | None
    cost: Cost | None
    model: str
    provider: str
