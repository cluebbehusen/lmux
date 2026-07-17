"""lmux-anthropic — Anthropic provider for lmux."""

from lmux_anthropic.auth import (
    AnthropicBedrockEnvAuthProvider,
    AnthropicBedrockSessionAuthProvider,
    AnthropicEnvAuthProvider,
    AnthropicFoundryEnvAuthProvider,
    AnthropicFoundryTokenAuthProvider,
    AnthropicVertexADCAuthProvider,
    AnthropicVertexServiceAccountAuthProvider,
)
from lmux_anthropic.cost import calculate_anthropic_cost
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import (
    AnthropicBedrockProvider,
    AnthropicFoundryProvider,
    AnthropicProvider,
    AnthropicVertexProvider,
)

__all__ = [
    "AnthropicBedrockEnvAuthProvider",
    "AnthropicBedrockProvider",
    "AnthropicBedrockSessionAuthProvider",
    "AnthropicEnvAuthProvider",
    "AnthropicFoundryEnvAuthProvider",
    "AnthropicFoundryProvider",
    "AnthropicFoundryTokenAuthProvider",
    "AnthropicParams",
    "AnthropicProvider",
    "AnthropicVertexADCAuthProvider",
    "AnthropicVertexProvider",
    "AnthropicVertexServiceAccountAuthProvider",
    "calculate_anthropic_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the HTTP client."""
    import httpx  # noqa: F401, PLC0415
