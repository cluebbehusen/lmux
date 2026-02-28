"""lmux-groq — Groq provider for lmux."""

from lmux_groq.auth import GroqEnvAuthProvider
from lmux_groq.cost import calculate_groq_cost
from lmux_groq.params import GroqParams
from lmux_groq.provider import GroqProvider

__all__ = [
    "GroqEnvAuthProvider",
    "GroqParams",
    "GroqProvider",
    "calculate_groq_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the Groq SDK."""
    import groq  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
