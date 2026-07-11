"""Pydantic wire models for Gemini REST response bodies.

Only the fields lmux consumes are declared; unknown fields are ignored (Pydantic's default),
so new server fields do not break parsing. Gemini uses camelCase on the wire, so a shared
alias generator maps snake_case field names to their camelCase aliases (``function_call`` ->
``functionCall``). A part is a field-bag (no ``type`` tag), so ``WirePart`` carries every part
shape lmux reads as optional fields and the mapper dispatches on which are present.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _GeminiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# MARK: generateContent (response + streaming chunk share this shape)


class WireFunctionCall(_GeminiModel):
    id: str | None = None
    name: str | None = None
    args: dict[str, Any] | None = None


class WireExecutableCode(_GeminiModel):
    code: str | None = None
    language: str | None = None


class WireCodeExecutionResult(_GeminiModel):
    outcome: str | None = None
    output: str | None = None


class WirePart(_GeminiModel):
    text: str | None = None
    thought: bool | None = None
    function_call: WireFunctionCall | None = None
    executable_code: WireExecutableCode | None = None
    code_execution_result: WireCodeExecutionResult | None = None


class WireContent(_GeminiModel):
    parts: list[WirePart] | None = None


class WireCandidate(_GeminiModel):
    content: WireContent | None = None
    finish_reason: str | None = None


class WireUsageMetadata(_GeminiModel):
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    cached_content_token_count: int | None = None
    thoughts_token_count: int | None = None


class WireGenerateContentResponse(_GeminiModel):
    candidates: list[WireCandidate] | None = None
    usage_metadata: WireUsageMetadata | None = None


# MARK: batchEmbedContents


class WireEmbedding(_GeminiModel):
    values: list[float] | None = None


class WireEmbeddingsMetadata(_GeminiModel):
    billable_character_count: int | None = None


class WireBatchEmbeddingsResponse(_GeminiModel):
    embeddings: list[WireEmbedding] | None = None
    metadata: WireEmbeddingsMetadata | None = None


# MARK: Vertex AI embeddings (:predict — instances/predictions shape)


class WireVertexEmbedStatistics(_GeminiModel):
    token_count: int = 0


class WireVertexEmbedding(_GeminiModel):
    values: list[float] = Field(default_factory=list)
    statistics: WireVertexEmbedStatistics = Field(default_factory=WireVertexEmbedStatistics)


class WireVertexPrediction(_GeminiModel):
    embeddings: WireVertexEmbedding = Field(default_factory=WireVertexEmbedding)


class WireVertexPredictResponse(_GeminiModel):
    predictions: list[WireVertexPrediction] = Field(default_factory=list)
