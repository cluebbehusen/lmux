"""Provider builders for the Google suite across offline / live / record modes.

Most scenarios authenticate with a plain API-key string — Vertex accepts ``x-goog-api-key`` on its
project-in-path endpoint, just like the Gemini Developer API — so offline replay uses the harness's
generic string stub and needs no bespoke credential stub. Two paths reject API keys and require ADC
(OAuth bearer): Vertex ``:embedContent`` and the ADC auth smoke test. Those use ``vertex_adc_provider``
(real ADC live/record) and, for the smoke test only, ``offline_adc_auth`` (a fake ``Credentials`` so
the bearer/quota-project headers are built offline without google.auth or a network refresh).

Vertex bakes the GCP project id into the URL path; offline builds the provider with the placeholder
project the cassette was normalized to, live/record with the real project.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import pytest

from lmux_google.auth import GoogleADCAuthProvider, GoogleAPIKeyAuthProvider
from lmux_google.provider import GoogleProvider

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


@pytest.fixture
def vertex_provider(vertex_project_placeholder: str) -> Callable[..., GoogleProvider]:
    """Build a Vertex (``vertexai=True``) provider for the active mode, API-key authenticated.

    Offline (``auth`` is the string stub): placeholder project so the endpoint-path assertion
    matches the normalized cassette. Live/record (``auth`` is None): the real ``VERTEXAI_API_KEY``
    and the real ``GOOGLE_CLOUD_PROJECT``.
    """

    def _build(auth: Any, transport: Any, *, location: str = "global", async_: bool = False) -> GoogleProvider:  # noqa: ANN401
        if auth is None:  # live / record
            resolved: Any = GoogleAPIKeyAuthProvider(env_var="VERTEXAI_API_KEY")
            project = os.environ["GOOGLE_CLOUD_PROJECT"]
        else:  # offline
            resolved = auth
            project = vertex_project_placeholder
        if async_:
            return GoogleProvider(
                auth=resolved, vertexai=True, project=project, location=location, async_transport=transport
            )
        return GoogleProvider(auth=resolved, vertexai=True, project=project, location=location, transport=transport)

    return _build


@pytest.fixture
def vertex_adc_provider(vertex_project_placeholder: str) -> Callable[..., GoogleProvider]:
    """Build a Vertex provider that authenticates with ADC (OAuth bearer) live/record — for the paths
    that reject API keys (:embedContent, and the auth smoke test). Offline uses the injected auth.
    """

    def _build(auth: Any, transport: Any, *, location: str = "global") -> GoogleProvider:  # noqa: ANN401
        if auth is None:  # live / record
            resolved: Any = GoogleADCAuthProvider()
            project = os.environ["GOOGLE_CLOUD_PROJECT"]
        else:  # offline
            resolved = auth
            project = vertex_project_placeholder
        return GoogleProvider(auth=resolved, vertexai=True, project=project, location=location, transport=transport)

    return _build


@pytest.fixture
def dev_provider() -> Callable[..., GoogleProvider]:
    """Build a Gemini Developer API (``vertexai=False``) provider for the active mode.

    Offline uses the injected string stub; live/record uses ``GEMINI_API_KEY``.
    """

    def _build(auth: Any, transport: Any, *, async_: bool = False) -> GoogleProvider:  # noqa: ANN401
        resolved: Any = auth if auth is not None else GoogleAPIKeyAuthProvider(env_var="GEMINI_API_KEY")
        if async_:
            return GoogleProvider(auth=resolved, vertexai=False, async_transport=transport)
        return GoogleProvider(auth=resolved, vertexai=False, transport=transport)

    return _build


class _OfflineCredentials:
    """A minimal google-auth Credentials stand-in: valid, so bearer_token never triggers a refresh."""

    valid = True
    token = "offline-token"  # noqa: S105 — dummy; the header is built offline but never sent
    quota_project_id = "lmux-integration"

    def refresh(self, _request: object) -> None:
        self.valid = True


class _OfflineADCAuth:
    """Offline auth for the ADC smoke test: returns fake Credentials so the provider builds the
    bearer + x-goog-user-project headers without google.auth or a network refresh.
    """

    def get_credentials(self) -> "Credentials":
        return cast("Credentials", _OfflineCredentials())

    async def aget_credentials(self) -> "Credentials":
        return cast("Credentials", _OfflineCredentials())


@pytest.fixture
def offline_adc_auth() -> _OfflineADCAuth:
    return _OfflineADCAuth()
