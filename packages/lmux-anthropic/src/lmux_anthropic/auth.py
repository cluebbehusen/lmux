"""Auth providers for the Anthropic API, Claude on Vertex AI, and Claude in Microsoft Foundry."""

import contextlib
import os
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lmux.exceptions import AuthenticationError
from lmux_anthropic._lazy import exchange_workload_identity_token

if TYPE_CHECKING:
    import boto3
    from google.auth.credentials import Credentials

_TOKEN_REFRESH_LEEWAY = 60.0


def _monotonic() -> float:
    """Return the monotonic clock, indirected so tests can pin the refresh clock."""
    return time.monotonic()


def _load_vertex_adc_credentials(scopes: list[str]) -> "tuple[Credentials, str | None]":
    try:
        import google.auth  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("[vertex] extra group is required for Vertex AI support") from e  # noqa: TRY003

    from google.auth import _cloud_sdk, environment_vars  # noqa: PLC0415
    from google.auth.credentials import with_scopes_if_required  # noqa: PLC0415

    cloud_sdk_file = _cloud_sdk.get_application_default_credentials_path()
    explicit_file = os.environ.get(environment_vars.CREDENTIALS)
    credentials_file = explicit_file or cloud_sdk_file
    if not os.path.isfile(credentials_file):
        credentials, project_id = google.auth.default(scopes=scopes)
        return cast("Credentials", credentials), project_id

    from lmux_anthropic._lazy import HttpxTransportRequest  # noqa: PLC0415

    request = HttpxTransportRequest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        credentials, project_id = google.auth.load_credentials_from_file(credentials_file, request=request)
    credentials = with_scopes_if_required(credentials, scopes)

    if credentials_file == cloud_sdk_file and not project_id:
        project_id = _cloud_sdk.get_project_id()

    explicit_project_id = os.environ.get(
        environment_vars.PROJECT,
        os.environ.get(environment_vars.LEGACY_PROJECT),
    )
    effective_project_id = explicit_project_id or project_id
    get_project_id = getattr(credentials, "get_project_id", None)
    if not effective_project_id and callable(get_project_id):
        effective_project_id = get_project_id(request=request)

    return cast("Credentials", credentials), cast("str | None", effective_project_id)


class AnthropicEnvAuthProvider:
    """Auth provider that reads the API key from the ANTHROPIC_API_KEY environment variable."""

    def get_credentials(self) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            msg = "ANTHROPIC_API_KEY environment variable is not set"
            raise AuthenticationError(msg, provider="anthropic")
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()


class AnthropicWorkloadIdentityAuthProvider:
    """Workload Identity Federation auth provider for the direct Anthropic API.

    Exchanges an IdP-issued OIDC identity token at ``POST /v1/oauth/token`` for a
    short-lived Anthropic access token and returns a callable that yields the current
    access token, re-exchanging shortly before it expires. The identity-token source
    is invoked on every exchange, since identity tokens carrying a ``jti`` claim
    (e.g. GitHub Actions) are accepted only once.

    Pass ``identity_token_provider`` as a callable returning a fresh identity token
    (e.g. one that calls AWS STS ``GetWebIdentityToken``), or ``identity_token_file``
    for a token projected to disk (e.g. a Kubernetes service-account token).

    The exchange targets the Anthropic API (``token_base_url``, default
    ``https://api.anthropic.com``) independently of the completion provider's
    ``base_url``, so a gateway that proxies only Messages traffic keeps working.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        federation_rule_id: str,
        organization_id: str,
        service_account_id: str | None = None,
        workspace_id: str | None = None,
        identity_token_provider: Callable[[], str] | None = None,
        identity_token_file: str | os.PathLike[str] | None = None,
        token_base_url: str | None = None,
    ) -> None:
        if identity_token_provider is not None and identity_token_file is not None:
            msg = "identity_token_provider and identity_token_file are mutually exclusive"
            raise ValueError(msg)
        if identity_token_provider is not None:
            self._identity_token_source: Callable[[], str] = identity_token_provider
        elif identity_token_file is not None:
            token_path = Path(identity_token_file)
            self._identity_token_source = lambda: token_path.read_text().strip()
        else:
            msg = "An identity_token_provider or identity_token_file is required"
            raise ValueError(msg)
        self._federation_rule_id: str = federation_rule_id
        self._organization_id: str = organization_id
        self._service_account_id: str | None = service_account_id
        self._workspace_id: str | None = workspace_id
        self._token_base_url: str | None = token_base_url
        self._access_token: str | None = None
        self._refresh_at: float = 0.0
        self._expires_at: float = 0.0
        self._refresh_lock: threading.Lock = threading.Lock()

    def _resolve_access_token(self) -> str:
        """Return the current access token, exchanging a fresh identity token when (nearly) expired.

        Refreshes are single-flight so concurrent callers never re-submit a single-use identity
        token. A caller with no valid token blocks until the winner's exchange completes and its
        failure propagates. In the advisory window (past the refresh point, before real expiry)
        callers never block: one attempts the refresh, preferring the still-valid cached token
        over a failure, while the rest proceed with the cached token immediately.
        """
        token = self._access_token
        if token is None or _monotonic() >= self._expires_at:
            with self._refresh_lock:
                token = self._access_token
                if token is None or _monotonic() >= self._expires_at:
                    token = self._refresh()
        elif _monotonic() >= self._refresh_at and self._refresh_lock.acquire(blocking=False):
            try:
                if _monotonic() >= self._refresh_at:
                    with contextlib.suppress(Exception):
                        token = self._refresh()
            finally:
                self._refresh_lock.release()
        return token

    def _refresh(self) -> str:
        access_token, expires_in = exchange_workload_identity_token(
            assertion=self._identity_token_source(),
            federation_rule_id=self._federation_rule_id,
            organization_id=self._organization_id,
            service_account_id=self._service_account_id,
            workspace_id=self._workspace_id,
            base_url=self._token_base_url,
        )
        now = _monotonic()
        self._access_token = access_token
        self._expires_at = now + expires_in
        self._refresh_at = now + expires_in - min(_TOKEN_REFRESH_LEEWAY, expires_in / 2)
        return access_token

    def get_credentials(self) -> Callable[[], str]:
        return self._resolve_access_token

    async def aget_credentials(self) -> Callable[[], str]:
        return self._resolve_access_token


class AnthropicVertexADCAuthProvider:
    """Default Vertex auth provider — uses Application Default Credentials.

    File-based credentials use lmux's httpx auth transport. If no file exists,
    ``google.auth.default()`` handles instance metadata and other environments.
    Returns the credentials together with the project ID that ADC resolved
    (which may be None). Requires the ``[vertex]`` extra.
    """

    def __init__(self, *, scopes: list[str] | None = None) -> None:
        self._scopes: list[str] = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "tuple[Credentials, str | None]":
        return _load_vertex_adc_credentials(self._scopes)

    async def aget_credentials(self) -> "tuple[Credentials, str | None]":
        return self.get_credentials()


class AnthropicVertexServiceAccountAuthProvider:
    """Vertex auth provider that loads credentials from a service account JSON key file.

    Accepts the file path to the JSON key file (the same value you would set
    in ``GOOGLE_APPLICATION_CREDENTIALS``). Returns the credentials together
    with the key file's project ID. Requires the ``[vertex]`` extra.
    """

    def __init__(
        self,
        *,
        service_account_file: str,
        scopes: list[str] | None = None,
    ) -> None:
        self._service_account_file: str = service_account_file
        self._scopes: list[str] = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "tuple[Credentials, str | None]":
        try:
            from google.oauth2 import service_account  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[vertex] extra group is required for Vertex AI support") from e  # noqa: TRY003

        credentials = service_account.Credentials.from_service_account_file(
            self._service_account_file, scopes=self._scopes
        )
        return credentials, cast("str | None", credentials.project_id)

    async def aget_credentials(self) -> "tuple[Credentials, str | None]":
        return self.get_credentials()


class AnthropicFoundryEnvAuthProvider:
    """Auth provider that reads the API key from the ANTHROPIC_FOUNDRY_API_KEY environment variable."""

    def get_credentials(self) -> str:
        api_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
        if api_key is None:
            msg = "ANTHROPIC_FOUNDRY_API_KEY environment variable is not set"
            raise AuthenticationError(msg, provider="anthropic-foundry")
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()


class AnthropicFoundryTokenAuthProvider:
    """Foundry auth provider that wraps a Microsoft Entra ID token provider.

    Pass a callable that returns a bearer token, e.g.
    ``azure.identity.get_bearer_token_provider(DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default")``. The SDK invokes the
    callable on every request.
    """

    def __init__(self, *, token_provider: Callable[[], str]) -> None:
        self._token_provider: Callable[[], str] = token_provider

    def get_credentials(self) -> Callable[[], str]:
        return self._token_provider

    async def aget_credentials(self) -> Callable[[], str]:
        return self._token_provider


class AnthropicBedrockEnvAuthProvider:
    """Default Bedrock auth provider — creates bare boto3 sessions that inherit from the environment.

    Credentials are resolved by boto3's default credential chain: environment variables
    (``AWS_BEARER_TOKEN_BEDROCK``, ``AWS_ACCESS_KEY_ID``, …), the default profile, instance
    metadata, etc. Requires the ``[bedrock]`` extra. SigV4 credentials are resolved synchronously
    even on the async request path, so ``aget_credentials`` returns the same sync session.
    """

    def get_credentials(self) -> "boto3.Session":
        import boto3  # noqa: PLC0415

        return boto3.Session()

    async def aget_credentials(self) -> "boto3.Session":
        return self.get_credentials()


class AnthropicBedrockSessionAuthProvider:
    """Bedrock auth provider that creates boto3 sessions with explicit configuration.

    Accepts the same keyword arguments as ``boto3.Session`` (``region_name``, ``profile_name``,
    ``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``). Requires the
    ``[bedrock]`` extra.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None:
        self._kwargs: dict[str, Any] = {
            "region_name": region_name,
            "profile_name": profile_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }

    def get_credentials(self) -> "boto3.Session":
        import boto3  # noqa: PLC0415

        return boto3.Session(**self._kwargs)

    async def aget_credentials(self) -> "boto3.Session":
        return self.get_credentials()
