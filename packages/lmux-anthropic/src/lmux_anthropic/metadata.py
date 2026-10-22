"""Response metadata for the Anthropic providers."""

from pydantic import BaseModel

from lmux.types import BaseProviderMetadata


class AnthropicStopDetails(BaseModel):
    """The ``stop_details`` object Anthropic reports when it refuses a request.

    ``category`` names the policy area (for example ``"cyber"`` or ``"bio"``) and
    ``explanation`` describes the refusal. Either can be ``None`` even on a refusal,
    so branch on ``finish_reason == "refusal"`` rather than on these fields.
    """

    type: str
    category: str | None = None
    explanation: str | None = None


class AnthropicMetadata(BaseProviderMetadata):
    """Anthropic-specific response metadata.

    Attached to ``ChatResponse.provider_metadata``, or to the terminal chunk when
    streaming, only when the API reports something here; otherwise ``provider_metadata``
    stays ``None``.
    """

    stop_details: AnthropicStopDetails | None = None
