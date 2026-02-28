"""Provider-specific parameters for Groq API calls."""

from typing import Literal

from lmux.types import BaseProviderParams


class GroqParams(BaseProviderParams):
    """Groq-specific parameters passed via ``provider_params``."""

    service_tier: Literal["auto", "on_demand", "flex", "performance"] | None = None
    seed: int | None = None
    user: str | None = None
