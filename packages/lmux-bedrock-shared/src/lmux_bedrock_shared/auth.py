"""Resolve AWS Bedrock request authentication: a Bedrock bearer token or SigV4 credentials.

Shared by lmux-aws-bedrock (Converse) and the native lmux-anthropic Bedrock provider. Both
resolve credentials once through boto3 (kept purely for the credential chain), then either attach
a bearer token from ``AWS_BEARER_TOKEN_BEDROCK`` or sign each request with SigV4 via
:func:`lmux_bedrock_shared.sign`. Credentials are frozen per request so a refreshable source
(SSO, assume-role, IMDS) never signs with a stale key.
"""

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NoReturn, Protocol

from lmux_bedrock_shared.sigv4 import sign

if TYPE_CHECKING:
    import boto3
    import httpx
    from botocore.credentials import Credentials

DEFAULT_REGION = "us-east-1"
BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105
_SERVICE = "bedrock"
_JSON_CONTENT_TYPE = "application/json"


class BedrockCredentialSource(Protocol):
    """A synchronous Bedrock credential source: a ``boto3.Session`` factory.

    Only ``get_credentials`` is needed to resolve a region and SigV4 credentials, so both Bedrock
    providers' auth providers satisfy this structurally — SigV4 credentials are resolved synchronously
    even on the async request path, so no async credential method is required here.
    """

    def get_credentials(self) -> "boto3.Session": ...


@dataclass(frozen=True)
class BedrockAuthContext:
    """Resolved per-provider auth: a region plus either a bearer token or a SigV4 credentials source."""

    region: str
    bearer_token: str | None = None
    credentials: "Credentials | None" = None

    def apply(self, request: "httpx.Request") -> None:
        """Attach the auth header(s) to a fully built request (bearer header, or SigV4 signature)."""
        if self.bearer_token is not None:
            request.headers["Authorization"] = f"Bearer {self.bearer_token}"
            return
        credentials = self.credentials
        if credentials is None:  # pragma: no cover - always set when bearer_token is None
            _raise_no_credentials()
        # Freeze credentials per request: a refreshable source (SSO, assume-role, IMDS) rotates the
        # keys over the provider's lifetime, so signing with a snapshot taken at client creation
        # would start failing once the original token expires.
        frozen = credentials.get_frozen_credentials()
        signed = sign(
            method=request.method,
            url=str(request.url),
            headers={"content-type": _JSON_CONTENT_TYPE},
            body=request.content,
            access_key=frozen.access_key or "",
            secret_key=frozen.secret_key or "",
            region=self.region,
            service=_SERVICE,
            now=datetime.now(UTC),
            session_token=frozen.token,
        )
        request.headers.update(signed)


def resolve_auth_context(auth: BedrockCredentialSource, region_override: str | None) -> BedrockAuthContext:
    """Resolve the auth mode once: bearer token if present, else a SigV4 credentials source.

    The region is resolved from the configured session (``AWS_DEFAULT_REGION``, a profile, or an
    explicit ``region_name``) so a bearer token reaches the right regional endpoint instead of
    silently defaulting to us-east-1. Bearer mode resolves the region best-effort and never *requires*
    a constructable session — a stale ``AWS_PROFILE`` must not break bearer auth, which does not use
    boto3 at all.
    """
    bearer = os.environ.get(BEARER_TOKEN_ENV)
    if bearer:
        return BedrockAuthContext(region=region_override or session_region(auth), bearer_token=bearer)

    session = auth.get_credentials()
    region = region_override or session.region_name or DEFAULT_REGION
    credentials = session.get_credentials()
    if credentials is None:
        _raise_no_credentials()
    return BedrockAuthContext(region=region, credentials=credentials)


def session_region(auth: BedrockCredentialSource) -> str:
    """Best-effort region from the configured session for bearer mode.

    ``region_name`` already reflects ``AWS_REGION``/``AWS_DEFAULT_REGION``/profile config; if the
    session cannot be constructed (e.g. a missing ``AWS_PROFILE``), fall back to the default region
    rather than failing a request that only needs a bearer token.
    """
    import botocore.exceptions  # noqa: PLC0415

    try:
        return auth.get_credentials().region_name or DEFAULT_REGION
    except botocore.exceptions.BotoCoreError:
        return DEFAULT_REGION


def _raise_no_credentials() -> NoReturn:
    """Raise botocore's ``NoCredentialsError`` so it maps to ``AuthenticationError``."""
    import botocore.exceptions  # noqa: PLC0415

    raise botocore.exceptions.NoCredentialsError
