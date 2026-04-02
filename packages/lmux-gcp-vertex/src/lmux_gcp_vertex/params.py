"""GCP Vertex AI provider-specific parameters."""

from typing import Any, Literal

from pydantic import BaseModel

from lmux.types import BaseProviderParams


class SafetySetting(BaseModel):
    """A single safety setting for content generation."""

    category: str
    threshold: str


class GoogleSearchTypes(BaseModel):
    """Search types to enable for Google Search."""

    web_search: bool | None = None
    image_search: bool | None = None


class GoogleSearchConfig(BaseModel):
    """Configuration for the Google Search tool."""

    search_types: GoogleSearchTypes | None = None
    exclude_domains: list[str] | None = None


class DynamicRetrievalConfig(BaseModel):
    """Configuration for dynamic retrieval behavior."""

    mode: Literal["MODE_UNSPECIFIED", "MODE_DYNAMIC"] | None = None
    dynamic_threshold: float | None = None


class GoogleSearchRetrievalConfig(BaseModel):
    """Configuration for the Google Search Retrieval tool."""

    dynamic_retrieval_config: DynamicRetrievalConfig | None = None


class GCPVertexParams(BaseProviderParams):
    """Vertex AI-specific parameters passed via ``provider_params``."""

    safety_settings: list[SafetySetting] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    labels: dict[str, str] | None = None
    thinking_config: dict[str, Any] | None = None
    google_search: GoogleSearchConfig | bool | None = None
    google_search_retrieval: GoogleSearchRetrievalConfig | None = None
    code_execution: bool | None = None
    task_type: str | None = None
