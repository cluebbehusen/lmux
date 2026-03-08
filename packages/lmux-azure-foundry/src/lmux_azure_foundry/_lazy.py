"""Lazy Azure OpenAI SDK loading internals.

Client creation is isolated here so tests can easily mock it
without patching sys.modules or using TYPE_CHECKING tricks.
"""

from typing import TYPE_CHECKING, Any

from lmux_azure_foundry.auth import AzureAdToken, AzureFoundryCredential

if TYPE_CHECKING:
    import openai


def _build_auth_kwargs(credential: AzureFoundryCredential) -> dict[str, Any]:
    """Convert a credential value to the appropriate AzureOpenAI constructor kwargs."""
    if isinstance(credential, AzureAdToken):
        return {"azure_ad_token": credential.token}
    if callable(credential):
        # Callable[[], str] — token provider function
        return {"azure_ad_token_provider": credential}
    # Plain str — API key
    return {"api_key": credential}


def create_sync_client(
    *,
    credential: AzureFoundryCredential,
    azure_endpoint: str,
    api_version: str,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "openai.AzureOpenAI":
    """Create an openai.AzureOpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "azure_endpoint": azure_endpoint,
        "api_version": api_version,
        **_build_auth_kwargs(credential),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return openai.AzureOpenAI(**kwargs)


def create_async_client(
    *,
    credential: AzureFoundryCredential,
    azure_endpoint: str,
    api_version: str,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "openai.AsyncAzureOpenAI":
    """Create an openai.AsyncAzureOpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "azure_endpoint": azure_endpoint,
        "api_version": api_version,
        **_build_auth_kwargs(credential),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return openai.AsyncAzureOpenAI(**kwargs)
