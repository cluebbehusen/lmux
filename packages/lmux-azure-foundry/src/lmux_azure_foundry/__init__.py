"""lmux-azure-foundry — Azure AI Foundry provider for lmux."""

from lmux_azure_foundry.auth import (
    AzureAdToken,
    AzureFoundryCredential,
    AzureFoundryKeyAuthProvider,
    AzureFoundryTokenAuthProvider,
)
from lmux_azure_foundry.cost import calculate_azure_foundry_cost
from lmux_azure_foundry.params import AzureFoundryParams
from lmux_azure_foundry.provider import AzureFoundryProvider

__all__ = [
    "AzureAdToken",
    "AzureFoundryCredential",
    "AzureFoundryKeyAuthProvider",
    "AzureFoundryParams",
    "AzureFoundryProvider",
    "AzureFoundryTokenAuthProvider",
    "calculate_azure_foundry_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import the OpenAI SDK.

    Call this during application startup to pay the import cost upfront
    rather than on the first request.
    """
    import openai  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
