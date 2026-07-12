"""HTTP client factories for the AWS Bedrock provider (SDK-lite, httpx transport).

Bedrock is reached over its REST endpoints (Converse / InvokeModel) with ``httpx``
directly rather than through boto3's client. boto3 is still used to resolve classic
AWS credentials for SigV4 signing (see ``provider``), but never sits in the request
path. Client creation is isolated here so tests can mock it without patching
``sys.modules``.
"""

from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx

DEFAULT_REGION = "us-east-1"


def bedrock_base_url(region: str, *, use_fips: bool = False) -> str:
    """Return the bedrock-runtime endpoint for a region, optionally the FIPS 140-3 variant.

    FIPS endpoints (``bedrock-runtime-fips.<region>.amazonaws.com``) are offered in the commercial
    and GovCloud regions where Bedrock runs; they force FIPS-validated in-transit cryptography.
    """
    service = "bedrock-runtime-fips" if use_fips else "bedrock-runtime"
    return f"https://{service}.{region}.amazonaws.com"


def create_sync_client(
    *,
    base_url: str,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Bedrock runtime endpoint.

    Auth headers are attached per request (SigV4 signs the exact body/host, or a
    bearer token is set), so the client carries no default ``Authorization``.
    """
    return _create_sync(base_url=base_url, headers={}, timeout=timeout, max_retries=max_retries)


def create_async_client(
    *,
    base_url: str,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Bedrock runtime endpoint."""
    return _create_async(base_url=base_url, headers={}, timeout=timeout, max_retries=max_retries)
