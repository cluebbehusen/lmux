"""Tests for Google (Gemini REST) wire models."""

from lmux_google._wire import WireGenerateContentResponse


class TestWireGenerateContentResponse:
    def test_only_continuation_parts_retain_unknown_fields(self) -> None:
        response = WireGenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thoughtSignature": "opaque",
                                    "futurePartField": {"kept": True},
                                    "functionCall": {"name": "weather", "futureCallField": 7},
                                }
                            ],
                            "futureContentField": "ignored",
                        },
                        "futureCandidateField": "ignored",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "futureUsageField": "ignored"},
                "futureResponseField": "ignored",
            }
        )

        assert response.model_dump(mode="json", by_alias=True, exclude_none=True) == {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "thoughtSignature": "opaque",
                                "functionCall": {"name": "weather", "futureCallField": 7},
                                "futurePartField": {"kept": True},
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 10},
        }
