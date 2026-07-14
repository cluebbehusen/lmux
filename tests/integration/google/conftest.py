"""Provider builders for the Google suite across offline / live / record modes.

The Google provider spans two backends. Both authenticate with a plain API-key string
(Vertex accepts ``x-goog-api-key`` on its project-in-path endpoint, just like the Gemini
Developer API), so offline replay uses the harness's generic string stub for both — no
bespoke credential stub is needed. Live/record supply the real key from the environment.

Vertex bakes the GCP project id into the URL path; offline builds the provider with the
placeholder project the cassette was normalized to, live/record with the real project.
"""

import os
from collections.abc import Callable
from typing import Any

import pytest

from lmux_google.auth import GoogleAPIKeyAuthProvider
from lmux_google.provider import GoogleProvider


@pytest.fixture
def vertex_provider(vertex_project_placeholder: str) -> Callable[..., GoogleProvider]:
    """Build a Vertex (``vertexai=True``) provider for the active mode.

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
