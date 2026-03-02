"""Authentication for Azure AI Foundry provider.

Azure AI Foundry (via the OpenAI SDK's ``AzureOpenAI`` class) supports three
authentication methods:

1. **API key** — passed as ``api_key`` to the SDK.
2. **Static Azure AD token** — passed as ``azure_ad_token``.
3. **Token provider callable** — passed as ``azure_ad_token_provider``; a
   ``() -> str`` callable that is invoked on every request to obtain a fresh
   token.  Typically created via
   ``azure.identity.get_bearer_token_provider(credential, scope)``.

The ``AzureFoundryCredential`` type alias represents any of these three, and
the provider dispatches to the correct SDK parameter at client creation time.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

from lmux.exceptions import AuthenticationError

PROVIDER = "azure-foundry"
DEFAULT_SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass(frozen=True, slots=True)
class AzureAdToken:
    """Wraps a static Azure AD / Entra ID token string.

    Used to distinguish a static token from an API key (both are plain
    strings) when passed through the ``AuthProvider`` protocol.
    """

    token: str


type AzureFoundryCredential = str | AzureAdToken | Callable[[], str]
"""Union of the three credential forms accepted by the Azure AI Foundry provider.

- ``str`` — API key.
- ``AzureAdToken`` — static Azure AD token.
- ``Callable[[], str]`` — token provider function (e.g. from
  ``get_bearer_token_provider``).
"""


class AzureFoundryKeyAuthProvider:
    """Auth provider that reads the API key from the AZURE_FOUNDRY_API_KEY environment variable."""

    def get_credentials(self) -> str:
        api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
        if api_key is None:
            msg = "AZURE_FOUNDRY_API_KEY environment variable is not set"
            raise AuthenticationError(msg, provider=PROVIDER)
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()


class AzureFoundryTokenAuthProvider:
    """Auth provider using Azure Identity ``DefaultAzureCredential``.

    Creates a token provider callable via ``get_bearer_token_provider`` that is
    suitable for ``AzureOpenAI(azure_ad_token_provider=...)``.

    Requires the ``identity`` extra: ``pip install lmux-azure-foundry[identity]``
    """

    def __init__(
        self,
        *,
        scopes: tuple[str, ...] = (DEFAULT_SCOPE,),
    ) -> None:
        self._scopes = scopes
        self._token_provider: Callable[[], str] | None = None

    def _ensure_token_provider(self) -> Callable[[], str]:
        if self._token_provider is None:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # noqa: PLC0415

            credential = DefaultAzureCredential()
            self._token_provider = get_bearer_token_provider(credential, *self._scopes)
        return self._token_provider

    def get_credentials(self) -> Callable[[], str]:
        return self._ensure_token_provider()

    async def aget_credentials(self) -> Callable[[], str]:
        return self._ensure_token_provider()
