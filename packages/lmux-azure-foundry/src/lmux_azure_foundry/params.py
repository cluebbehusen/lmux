"""Azure AI Foundry-specific provider parameters."""

from typing import Literal

from lmux.types import BaseProviderParams


class AzureFoundryParams(BaseProviderParams):
    """Provider-specific parameters for Azure AI Foundry API calls."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
    prompt_cache_key: str | None = None
    """Cache key for Azure's automatic prompt caching, scoping cache hits to requests that share it.

    Azure Foundry caches long prompt prefixes implicitly; this key partitions that cache (e.g. per
    tenant or per prompt template) and replaces ``user`` for cache routing. Sent on Chat Completions only.
    Azure does not support OpenAI's explicit cache breakpoints, so ``CachePointContent`` is dropped.
    """
    prompt_cache_retention: str | None = None
    """Retention policy for the prompt cache, passed through as Azure's ``prompt_cache_retention``.

    Sent on Chat Completions only.
    """
    deployment_type: Literal["global", "data_zone", "regional"] | None = None
    """Deployment type for cost calculation.

    - ``None`` / ``"global"`` — Global Standard pricing (default, no multiplier).
    - ``"data_zone"`` — Data Zone deployment (1.1x multiplier).
    - ``"regional"`` — Regional deployment (~1.1x multiplier).

    This parameter only affects cost calculation and is **not** sent to the API.
    """
