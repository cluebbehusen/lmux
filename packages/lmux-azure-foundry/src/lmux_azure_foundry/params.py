"""Azure AI Foundry-specific provider parameters."""

from typing import Literal

from lmux.types import BaseProviderParams


class AzureFoundryParams(BaseProviderParams):
    """Provider-specific parameters for Azure AI Foundry API calls."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
