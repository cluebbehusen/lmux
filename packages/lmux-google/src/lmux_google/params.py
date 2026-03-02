"""Google Vertex AI provider-specific parameters."""

from typing import Any

from pydantic import BaseModel

from lmux.types import BaseProviderParams


class SafetySetting(BaseModel):
    """A safety filter setting for content generation.

    ``category`` values: ``HARM_CATEGORY_HARASSMENT``, ``HARM_CATEGORY_HATE_SPEECH``,
    ``HARM_CATEGORY_SEXUALLY_EXPLICIT``, ``HARM_CATEGORY_DANGEROUS_CONTENT``, etc.

    ``threshold`` values: ``BLOCK_LOW_AND_ABOVE``, ``BLOCK_MEDIUM_AND_ABOVE``,
    ``BLOCK_ONLY_HIGH``, ``BLOCK_NONE``, ``OFF``.
    """

    category: str
    threshold: str


class GoogleParams(BaseProviderParams):
    """Google Vertex AI-specific parameters for chat completion."""

    safety_settings: list[SafetySetting] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    labels: dict[str, str] | None = None
    thinking_config: dict[str, Any] | None = None
