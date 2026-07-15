"""Provider builder for the Azure AI Foundry suite across offline / live / record modes.

Azure Foundry authenticates with a plain API-key string, so offline replay uses the harness's
generic string stub. Live/record read the key from ``AZURE_FOUNDRY_KEY`` and the resource endpoint
from ``AZURE_FOUNDRY_ENDPOINT``. The resource name lives in the endpoint host; offline builds the
provider against the placeholder host the cassette was normalized to.
"""

import os
from collections.abc import Callable
from typing import Any

import pytest

from lmux_azure_foundry.auth import AzureAdToken, AzureFoundryTokenAuthProvider
from lmux_azure_foundry.provider import AzureFoundryProvider


class _EnvKeyAuth:
    """Live/record auth: the Foundry API key from ``AZURE_FOUNDRY_KEY`` (the provider's default reads a
    different env var, so this keeps the test aligned with the maintainer's key)."""

    def get_credentials(self) -> str:
        return os.environ["AZURE_FOUNDRY_KEY"]

    async def aget_credentials(self) -> str:
        return os.environ["AZURE_FOUNDRY_KEY"]


@pytest.fixture
def foundry_provider(azure_resource_placeholder: str) -> Callable[..., AzureFoundryProvider]:
    """Build an Azure Foundry provider for the active mode.

    Offline (``auth`` is the string stub): the placeholder resource host so the endpoint-path
    assertion matches the normalized cassette. Live/record (``auth`` is None): the real
    ``AZURE_FOUNDRY_KEY`` and ``AZURE_FOUNDRY_ENDPOINT``.
    """

    def _build(auth: Any, transport: Any, *, async_: bool = False) -> AzureFoundryProvider:  # noqa: ANN401
        if auth is None:  # live / record
            endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"]
            resolved: Any = _EnvKeyAuth()
        else:  # offline
            endpoint = f"https://{azure_resource_placeholder}.openai.azure.com"
            resolved = auth
        if async_:
            return AzureFoundryProvider(endpoint=endpoint, auth=resolved, async_transport=transport)
        return AzureFoundryProvider(endpoint=endpoint, auth=resolved, transport=transport)

    return _build


class _OfflineAdAuth:
    """Offline auth for the AD smoke test: a static ``AzureAdToken`` so the provider builds the
    ``Authorization: Bearer`` headers offline without Azure Identity or a network token fetch."""

    def get_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="offline-token")  # noqa: S106 — dummy; the header is built offline but never sent

    async def aget_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="offline-token")  # noqa: S106


@pytest.fixture
def offline_ad_auth() -> _OfflineAdAuth:
    return _OfflineAdAuth()


@pytest.fixture
def foundry_ad_provider(azure_resource_placeholder: str) -> Callable[..., AzureFoundryProvider]:
    """Build a provider that authenticates with Azure AD (Entra) bearer tokens live/record via
    ``DefaultAzureCredential``. Offline uses the injected fake-token auth to exercise the bearer path."""

    def _build(auth: Any, transport: Any) -> AzureFoundryProvider:  # noqa: ANN401
        if auth is None:  # live / record
            return AzureFoundryProvider(
                endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"], auth=AzureFoundryTokenAuthProvider(), transport=transport
            )
        endpoint = f"https://{azure_resource_placeholder}.openai.azure.com"
        return AzureFoundryProvider(endpoint=endpoint, auth=auth, transport=transport)

    return _build
