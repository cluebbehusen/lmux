"""lmux-google — Google (Gemini) provider for lmux."""

from lmux_google.auth import (
    GoogleADCAuthProvider,
    GoogleAPIKeyAuthProvider,
    GoogleServiceAccountAuthProvider,
)
from lmux_google.cost import calculate_google_cost
from lmux_google.params import (
    DynamicRetrievalConfig,
    GoogleParams,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
)
from lmux_google.provider import GoogleProvider

__all__ = [
    "DynamicRetrievalConfig",
    "GoogleADCAuthProvider",
    "GoogleAPIKeyAuthProvider",
    "GoogleParams",
    "GoogleProvider",
    "GoogleSearchConfig",
    "GoogleSearchRetrievalConfig",
    "GoogleSearchTypes",
    "GoogleServiceAccountAuthProvider",
    "SafetySetting",
    "calculate_google_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the HTTP client and google-auth."""
    import google.auth  # noqa: PLC0415, F401
    import httpx  # noqa: PLC0415, F401
