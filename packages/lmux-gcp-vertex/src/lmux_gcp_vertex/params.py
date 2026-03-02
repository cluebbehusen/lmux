"""GCP Vertex AI provider-specific parameters."""

from typing import Any

from pydantic import BaseModel

from lmux.types import BaseProviderParams


class SafetySetting(BaseModel):
    """A single safety setting for content generation."""

    category: str
    threshold: str


class GCPVertexParams(BaseProviderParams):
    """Vertex AI-specific parameters passed via ``provider_params``."""

    safety_settings: list[SafetySetting] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    labels: dict[str, str] | None = None
    thinking_config: dict[str, Any] | None = None
