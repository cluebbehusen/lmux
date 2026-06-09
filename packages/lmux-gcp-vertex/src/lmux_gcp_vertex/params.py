"""Deprecated shim — use ``lmux_google.params`` instead."""

from lmux_google.params import (
    DynamicRetrievalConfig,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
)
from lmux_google.params import (
    GoogleParams as GCPVertexParams,
)

__all__ = [
    "DynamicRetrievalConfig",
    "GCPVertexParams",
    "GoogleSearchConfig",
    "GoogleSearchRetrievalConfig",
    "GoogleSearchTypes",
    "SafetySetting",
]
