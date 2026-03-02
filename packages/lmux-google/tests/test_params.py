"""Tests for Google Vertex AI provider-specific parameters."""

from lmux_google.params import GoogleParams, SafetySetting


class TestSafetySetting:
    def test_creation(self) -> None:
        setting = SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE")
        assert setting.category == "HARM_CATEGORY_HARASSMENT"
        assert setting.threshold == "BLOCK_LOW_AND_ABOVE"


class TestGoogleParams:
    def test_defaults(self) -> None:
        params = GoogleParams()
        assert params.safety_settings is None
        assert params.presence_penalty is None
        assert params.frequency_penalty is None
        assert params.seed is None
        assert params.labels is None
        assert params.thinking_config is None

    def test_all_fields(self) -> None:
        params = GoogleParams(
            safety_settings=[SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")],
            presence_penalty=0.5,
            frequency_penalty=0.3,
            seed=42,
            labels={"env": "test"},
            thinking_config={"thinking_budget": 1024},
        )
        assert params.safety_settings is not None
        assert len(params.safety_settings) == 1
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == 0.3
        assert params.seed == 42
        assert params.labels == {"env": "test"}
        assert params.thinking_config == {"thinking_budget": 1024}
