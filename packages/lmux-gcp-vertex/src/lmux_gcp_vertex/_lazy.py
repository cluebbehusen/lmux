"""Lazy google-genai SDK loading internals.

Client creation is isolated here so tests can easily mock it
without patching sys.modules or using TYPE_CHECKING tricks.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai import Client


def create_client(
    *,
    vertexai: bool = True,
    project: str | None = None,
    location: str | None = None,
    credentials: "Credentials | None" = None,
    api_key: str | None = None,
) -> "Client":
    """Create a google-genai Client."""
    from google import genai  # noqa: PLC0415

    kwargs: dict[str, Any] = {"vertexai": vertexai}
    if project is not None:
        kwargs["project"] = project
    if location is not None:
        kwargs["location"] = location
    if credentials is not None:
        kwargs["credentials"] = credentials
    if api_key is not None:
        kwargs["api_key"] = api_key
    return genai.Client(**kwargs)
