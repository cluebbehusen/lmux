"""Deprecated shim — use ``lmux_google.auth`` instead."""

from lmux_google.auth import (
    PROVIDER_NAME,
)
from lmux_google.auth import (
    GoogleADCAuthProvider as GCPVertexADCAuthProvider,
)
from lmux_google.auth import (
    GoogleAPIKeyAuthProvider as GCPVertexAPIKeyAuthProvider,
)
from lmux_google.auth import (
    GoogleServiceAccountAuthProvider as GCPVertexServiceAccountAuthProvider,
)

__all__ = [
    "PROVIDER_NAME",
    "GCPVertexADCAuthProvider",
    "GCPVertexAPIKeyAuthProvider",
    "GCPVertexServiceAccountAuthProvider",
]
