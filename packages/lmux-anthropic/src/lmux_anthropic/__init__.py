"""lmux-anthropic — Anthropic provider for lmux."""

from lmux_anthropic.auth import (
    AnthropicEnvAuthProvider,
    AnthropicVertexADCAuthProvider,
    AnthropicVertexServiceAccountAuthProvider,
)
from lmux_anthropic.cost import calculate_anthropic_cost
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import AnthropicProvider, AnthropicVertexProvider

__all__ = [
    "AnthropicEnvAuthProvider",
    "AnthropicParams",
    "AnthropicProvider",
    "AnthropicVertexADCAuthProvider",
    "AnthropicVertexProvider",
    "AnthropicVertexServiceAccountAuthProvider",
    "calculate_anthropic_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the Anthropic SDK."""
    import anthropic  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
