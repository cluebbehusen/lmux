"""Provider-specific parameters for Groq API calls."""

from datetime import date
from typing import Literal

from lmux.types import BaseProviderParams


class GroqParams(BaseProviderParams):
    """Groq-specific parameters passed via ``provider_params``."""

    service_tier: Literal["auto", "on_demand", "flex", "performance"] | None = None
    reasoning_effort: Literal["none", "default", "low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
    # Override the date used to resolve dated pricing (e.g. a scheduled rate
    # change); defaults to the current date when unset.
    pricing_as_of: date | None = None
