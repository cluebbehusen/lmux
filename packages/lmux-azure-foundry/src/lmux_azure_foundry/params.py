"""Azure AI Foundry-specific provider parameters."""

from typing import Literal

from lmux.types import BaseProviderParams


class AzureFoundryParams(BaseProviderParams):
    """Provider-specific parameters for Azure AI Foundry API calls."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
    deployment_type: Literal["global", "data_zone", "regional"] | None = None
    """Deployment type for cost calculation.

    - ``None`` / ``"global"`` — Global Standard pricing (default, no multiplier).
    - ``"data_zone"`` — Data Zone deployment (1.1x multiplier).
    - ``"regional"`` — Regional deployment (~1.1x multiplier).

    This parameter only affects cost calculation and is **not** sent to the API.
    """
