"""lmux-openai — OpenAI provider for lmux."""

from lmux_openai.auth import OpenAIEnvAuthProvider
from lmux_openai.cost import PRICING as OPENAI_PRICING
from lmux_openai.cost import calculate_openai_cost
from lmux_openai.params import OpenAIParams
from lmux_openai.provider import OpenAIProvider

__all__ = [
    "OPENAI_PRICING",
    "OpenAIEnvAuthProvider",
    "OpenAIParams",
    "OpenAIProvider",
    "calculate_openai_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the OpenAI SDK.

    Call this during application startup to pay the import cost upfront
    rather than on the first request.
    """
    import openai  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
