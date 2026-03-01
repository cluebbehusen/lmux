"""AWS Bedrock-specific provider parameters."""

from typing import Any, Literal

from pydantic import BaseModel

from lmux.types import BaseProviderParams


class GuardrailConfig(BaseModel):
    """Configuration for a Bedrock guardrail applied to the request."""

    guardrail_identifier: str
    guardrail_version: str
    trace: Literal["enabled", "disabled", "enabled_full"] | None = None


class BedrockParams(BaseProviderParams):
    """Provider-specific parameters for AWS Bedrock API calls."""

    guardrail_config: GuardrailConfig | None = None
    additional_model_request_fields: dict[str, Any] | None = None
    additional_model_response_field_paths: list[str] | None = None
