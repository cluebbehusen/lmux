"""Deprecated shim — use ``lmux_google.provider`` instead."""

from lmux_google.provider import (
    PROVIDER_NAME,
)
from lmux_google.provider import (
    GoogleAuth as GCPVertexAuth,
)
from lmux_google.provider import (
    GoogleProvider as GCPVertexProvider,
)

__all__ = [
    "PROVIDER_NAME",
    "GCPVertexAuth",
    "GCPVertexProvider",
]
