"""lmux-gcp-vertex — DEPRECATED; renamed to lmux-google.

This package is a compatibility shim that re-exports lmux-google under the
old names. Install ``lmux-google`` and import from ``lmux_google`` instead.
"""

import warnings

from lmux_google import (
    DynamicRetrievalConfig,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
    preload,
)
from lmux_google import (
    GoogleADCAuthProvider as GCPVertexADCAuthProvider,
)
from lmux_google import (
    GoogleAPIKeyAuthProvider as GCPVertexAPIKeyAuthProvider,
)
from lmux_google import (
    GoogleParams as GCPVertexParams,
)
from lmux_google import (
    GoogleProvider as GCPVertexProvider,
)
from lmux_google import (
    GoogleServiceAccountAuthProvider as GCPVertexServiceAccountAuthProvider,
)
from lmux_google import (
    calculate_google_cost as calculate_gcp_vertex_cost,
)

__all__ = [
    "DynamicRetrievalConfig",
    "GCPVertexADCAuthProvider",
    "GCPVertexAPIKeyAuthProvider",
    "GCPVertexParams",
    "GCPVertexProvider",
    "GCPVertexServiceAccountAuthProvider",
    "GoogleSearchConfig",
    "GoogleSearchRetrievalConfig",
    "GoogleSearchTypes",
    "SafetySetting",
    "calculate_gcp_vertex_cost",
    "preload",
]

warnings.warn(
    "lmux-gcp-vertex has been renamed to lmux-google; install lmux-google and import from lmux_google instead.",
    DeprecationWarning,
    stacklevel=2,
)
