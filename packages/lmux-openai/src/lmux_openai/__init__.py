"""lmux-openai — OpenAI provider for lmux."""

from lmux_openai.auth import OpenAIEnvAuthProvider
from lmux_openai.cost import REGIONAL_UPLIFT, apply_cost_multiplier, calculate_openai_cost, regional_uplift_applies
from lmux_openai.params import OpenAIParams
from lmux_openai.provider import OpenAIProvider

__all__ = [
    "REGIONAL_UPLIFT",
    "OpenAIEnvAuthProvider",
    "OpenAIParams",
    "OpenAIProvider",
    "apply_cost_multiplier",
    "calculate_openai_cost",
    "preload",
    "regional_uplift_applies",
]


def preload() -> None:
    """Eagerly import the OpenAI SDK.

    Call this during application startup to pay the import cost upfront
    rather than on the first request.
    """
    import openai  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
