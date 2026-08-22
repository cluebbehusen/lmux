"""Authentication for the Google provider.

The simplest way to authenticate is to use Application Default Credentials (ADC)
via ``GoogleADCAuthProvider`` (the default).  ADC searches for credentials in:

1. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable (path to a Google
   credential configuration file)
2. ``gcloud auth application-default login`` cached credentials
3. Compute Engine / Cloud Run / GKE metadata server

See: https://cloud.google.com/docs/authentication/application-default-credentials

For API key authentication, use ``GoogleAPIKeyAuthProvider``.

See: https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
"""

import os
import warnings
from typing import TYPE_CHECKING, cast

from lmux.exceptions import AuthenticationError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

PROVIDER_NAME = "google"


def _load_adc_credentials(scopes: list[str]) -> "Credentials":
    import google.auth  # noqa: PLC0415
    from google.auth import _cloud_sdk, environment_vars  # noqa: PLC0415
    from google.auth.credentials import with_scopes_if_required  # noqa: PLC0415

    cloud_sdk_file = _cloud_sdk.get_application_default_credentials_path()
    explicit_file = os.environ.get(environment_vars.CREDENTIALS)
    credentials_file = explicit_file or cloud_sdk_file
    if not os.path.isfile(credentials_file):
        credentials, _ = google.auth.default(scopes=scopes)
        return cast("Credentials", credentials)

    from lmux_google._lazy import HttpxAuthRequest  # noqa: PLC0415

    request = HttpxAuthRequest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        credentials, _ = google.auth.load_credentials_from_file(credentials_file, request=request)
    credentials = with_scopes_if_required(credentials, scopes)
    return cast("Credentials", credentials)


class GoogleADCAuthProvider:
    """Default auth provider — uses Application Default Credentials.

    File-based credentials use lmux's httpx auth transport. If no file exists,
    ``google.auth.default()`` handles instance metadata and other environments.

    Scopes default to ``cloud-platform`` which is required for Vertex AI
    and for Workload Identity Federation impersonation flows.
    """

    def __init__(self, *, scopes: list[str] | None = None) -> None:
        self._scopes: list[str] = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "Credentials":
        return _load_adc_credentials(self._scopes)

    async def aget_credentials(self) -> "Credentials":
        return self.get_credentials()


class GoogleServiceAccountAuthProvider:
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
        self._service_account_file: str = service_account_file
        self._scopes: list[str] = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "Credentials":
        from google.oauth2 import service_account  # noqa: PLC0415

        return service_account.Credentials.from_service_account_file(self._service_account_file, scopes=self._scopes)

    async def aget_credentials(self) -> "Credentials":
        from google.oauth2 import service_account  # noqa: PLC0415

        return service_account.Credentials.from_service_account_file(self._service_account_file, scopes=self._scopes)


class GoogleAPIKeyAuthProvider:
    """Auth provider that uses a Google Cloud API key.

    Reads from the ``GOOGLE_API_KEY`` environment variable by default,
    or accepts a key directly.
    """

    def __init__(self, *, api_key: str | None = None, env_var: str = "GOOGLE_API_KEY") -> None:
        self._api_key: str | None = api_key
        self._env_var: str = env_var

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
