"""Tests for AWS Bedrock provider parameters."""

from lmux_aws_bedrock.params import BedrockParams, GuardrailConfig


class TestGuardrailConfig:
    def test_required_fields(self) -> None:
        config = GuardrailConfig(guardrail_identifier="abc", guardrail_version="1")
        assert config.guardrail_identifier == "abc"
        assert config.guardrail_version == "1"
        assert config.trace is None

    def test_with_trace(self) -> None:
        config = GuardrailConfig(guardrail_identifier="abc", guardrail_version="1", trace="enabled")
        assert config.trace == "enabled"


class TestBedrockParams:
    def test_defaults(self) -> None:
        params = BedrockParams()
        assert params.guardrail_config is None
        assert params.additional_model_request_fields is None
        assert params.additional_model_response_field_paths is None

    def test_with_guardrail_config(self) -> None:
        config = GuardrailConfig(guardrail_identifier="abc", guardrail_version="1")
        params = BedrockParams(guardrail_config=config)
        assert params.guardrail_config == config

    def test_with_additional_model_request_fields(self) -> None:
        fields = {"top_k": 50}
        params = BedrockParams(additional_model_request_fields=fields)
        assert params.additional_model_request_fields == fields

    def test_with_additional_model_response_field_paths(self) -> None:
        paths = ["stop_sequence"]
        params = BedrockParams(additional_model_response_field_paths=paths)
        assert params.additional_model_response_field_paths == paths
