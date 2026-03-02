"""lmux-google — Google Vertex AI provider for lmux."""

from lmux_google.auth import GoogleADCAuthProvider, GoogleServiceAccountAuthProvider
from lmux_google.cost import calculate_google_cost
from lmux_google.params import GoogleParams, SafetySetting
from lmux_google.provider import GoogleProvider

__all__ = [
    "GoogleADCAuthProvider",
    "GoogleParams",
    "GoogleProvider",
    "GoogleServiceAccountAuthProvider",
    "SafetySetting",
    "calculate_google_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the google-genai SDK."""
    import google.genai  # noqa: PLC0415, F401  # pyright: ignore[reportUnusedImport]
