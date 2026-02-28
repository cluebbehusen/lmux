"""OpenAI-specific provider parameters."""

from typing import Literal

from pydantic import BaseModel


class OpenAIParams(BaseModel):
    """Provider-specific parameters for OpenAI API calls."""

    service_tier: Literal["auto", "default", "flex"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
