"""Provider-specific parameters for Anthropic API calls."""

from typing import Literal

from pydantic import BaseModel


class AnthropicParams(BaseModel):
    """Anthropic-specific parameters passed via ``provider_params``."""

    thinking: dict[str, object] | None = None
    metadata: dict[str, str] | None = None
    top_k: int | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    inference_geo: Literal["us"] | None = None
    speed: Literal["fast"] | None = None
