"""Authentication for Google Cloud Vertex AI provider.

The simplest way to authenticate is to use Application Default Credentials (ADC)
via ``GCPVertexADCAuthProvider`` (the default).  ADC searches for credentials in:

1. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable (path to a service
   account JSON key file)
2. ``gcloud auth application-default login`` cached credentials
3. Compute Engine / Cloud Run / GKE metadata server

See: https://cloud.google.com/docs/authentication/application-default-credentials

For API key authentication, use ``GCPVertexAPIKeyAuthProvider``.

See: https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
"""

import os
from typing import TYPE_CHECKING, cast

from lmux.exceptions import AuthenticationError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

PROVIDER_NAME = "gcp-vertex"


class GCPVertexADCAuthProvider:
    """Default auth provider — uses Application Default Credentials.

    Credentials are resolved by ``google.auth.default()`` which searches
    environment variables, ``gcloud`` CLI defaults, and instance metadata.
    """

    def get_credentials(self) -> "Credentials":
        import google.auth  # noqa: PLC0415

        # google.auth has unresolvable string forward-ref annotations; cast is required
        return cast("Credentials", google.auth.default()[0])

    async def aget_credentials(self) -> "Credentials":
        import google.auth  # noqa: PLC0415

        return cast("Credentials", google.auth.default()[0])


class GCPVertexServiceAccountAuthProvider:
    """Auth provider that loads credentials from a service account JSON key file.

    Accepts the file path to the JSON key file (the same value you would set
    in ``GOOGLE_APPLICATION_CREDENTIALS``).
    """

    def __init__(
        self,
        *,
        service_account_file: str,
        scopes: list[str] | None = None,
    ) -> None:
        self._service_account_file = service_account_file
        self._scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "Credentials":
        from google.oauth2 import service_account  # noqa: PLC0415

        return service_account.Credentials.from_service_account_file(self._service_account_file, scopes=self._scopes)

    async def aget_credentials(self) -> "Credentials":
        from google.oauth2 import service_account  # noqa: PLC0415

        return service_account.Credentials.from_service_account_file(self._service_account_file, scopes=self._scopes)


class GCPVertexAPIKeyAuthProvider:
    """Auth provider that uses a Google Cloud API key.

    Reads from the ``GOOGLE_API_KEY`` environment variable by default,
    or accepts a key directly.
    """

    def __init__(self, *, api_key: str | None = None, env_var: str = "GOOGLE_API_KEY") -> None:
        self._api_key = api_key
        self._env_var = env_var

    def get_credentials(self) -> str:
        if self._api_key is not None:
            return self._api_key
        api_key = os.environ.get(self._env_var)
        if api_key is None:
            msg = f"{self._env_var} environment variable is not set"
            raise AuthenticationError(msg, provider=PROVIDER_NAME)
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()
