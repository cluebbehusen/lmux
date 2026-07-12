"""OpenAI-specific provider parameters."""

from typing import Literal

from lmux.types import BaseProviderParams


class OpenAIParams(BaseProviderParams):
    """Provider-specific parameters for OpenAI API calls."""

    service_tier: Literal["auto", "default", "flex"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    user: str | None = None
    prompt_cache_key: str | None = None
    """Routing key for prompt caching, combined with the prefix hash to improve cache hit rates.

    Recommended (not required) for gpt-5.6+ to get reliable prefix matching across load-balanced
    backends, for both implicit and explicit caching. Applies to Chat Completions and the Responses API.
    """
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    """Prompt-cache retention window. Legacy: applies to pre-gpt-5.6 models; gpt-5.6+ use
    ``prompt_cache_options.ttl`` instead. Applies to Chat Completions and the Responses API.
    """
