"""Provider-specific parameters for Anthropic API calls."""

from datetime import date
from typing import Literal

from lmux.types import BaseProviderParams


class AnthropicParams(BaseProviderParams):
    """Anthropic-specific parameters passed via ``provider_params``."""

    thinking: dict[str, object] | None = None
    metadata: dict[str, str] | None = None
    top_k: int | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    inference_geo: Literal["us"] | None = None
    cache_control: dict[str, object] | None = None
    # Override the date used to resolve dated pricing (e.g. a model's
    # introductory-rate window); defaults to the current date when unset.
    pricing_as_of: date | None = None
