"""Auth providers for the Anthropic API, Claude on Vertex AI, and Claude in Microsoft Foundry."""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from lmux.exceptions import AuthenticationError

if TYPE_CHECKING:
    import boto3
    from google.auth.credentials import Credentials


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


class AnthropicVertexADCAuthProvider:
    """Default Vertex auth provider — uses Application Default Credentials.

    Credentials are resolved by ``google.auth.default()`` which searches
    environment variables, ``gcloud`` CLI defaults, and instance metadata.
    Returns the credentials together with the project ID that ADC resolved
    (which may be None). Requires the ``[vertex]`` extra.
    """

    def __init__(self, *, scopes: list[str] | None = None) -> None:
        self._scopes: list[str] = scopes or ["https://www.googleapis.com/auth/cloud-platform"]

    def get_credentials(self) -> "tuple[Credentials, str | None]":
        try:
            import google.auth  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[vertex] extra group is required for Vertex AI support") from e  # noqa: TRY003

        credentials, project_id = google.auth.default(scopes=self._scopes)
        # google.auth has unresolvable string forward-ref annotations; cast is required
        return cast("Credentials", credentials), project_id

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
