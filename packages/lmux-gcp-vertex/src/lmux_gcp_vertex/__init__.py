"""lmux-gcp-vertex — Google Cloud Vertex AI provider for lmux."""

from lmux_gcp_vertex.auth import (
    GCPVertexADCAuthProvider,
    GCPVertexAPIKeyAuthProvider,
    GCPVertexServiceAccountAuthProvider,
)
from lmux_gcp_vertex.cost import calculate_gcp_vertex_cost
from lmux_gcp_vertex.params import (
    DynamicRetrievalConfig,
    GCPVertexParams,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
)
from lmux_gcp_vertex.provider import GCPVertexProvider

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


def preload() -> None:
    """Eagerly import the google-genai SDK."""
    import google.genai  # noqa: PLC0415, F401  # pyright: ignore[reportUnusedImport]
