"""Provider-specific parameters for Anthropic API calls."""

from typing import Literal

from lmux.types import BaseProviderParams


class AnthropicParams(BaseProviderParams):
    """Anthropic-specific parameters passed via ``provider_params``."""

    thinking: dict[str, object] | None = None
    metadata: dict[str, str] | None = None
    top_k: int | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    inference_geo: Literal["us"] | None = None
    speed: Literal["fast"] | None = None
