"""Tests for GCP Vertex AI provider-specific parameters."""

from lmux_gcp_vertex.params import (
    DynamicRetrievalConfig,
    GCPVertexParams,
    GoogleSearchConfig,
    GoogleSearchRetrievalConfig,
    GoogleSearchTypes,
    SafetySetting,
)


class TestSafetySetting:
    def test_creation(self) -> None:
        setting = SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE")
        assert setting.category == "HARM_CATEGORY_HARASSMENT"
        assert setting.threshold == "BLOCK_LOW_AND_ABOVE"


class TestGoogleSearchTypes:
    def test_defaults(self) -> None:
        types = GoogleSearchTypes()
        assert types.web_search is None
        assert types.image_search is None

    def test_all_fields(self) -> None:
        types = GoogleSearchTypes(web_search=True, image_search=True)
        assert types.web_search is True
        assert types.image_search is True


class TestGoogleSearchConfig:
    def test_defaults(self) -> None:
        config = GoogleSearchConfig()
        assert config.search_types is None
        assert config.exclude_domains is None

    def test_all_fields(self) -> None:
        config = GoogleSearchConfig(
            search_types=GoogleSearchTypes(web_search=True),
            exclude_domains=["example.com"],
        )
        assert config.search_types is not None
        assert config.search_types.web_search is True
        assert config.exclude_domains == ["example.com"]


class TestDynamicRetrievalConfig:
    def test_defaults(self) -> None:
        config = DynamicRetrievalConfig()
        assert config.mode is None
        assert config.dynamic_threshold is None

    def test_all_fields(self) -> None:
        config = DynamicRetrievalConfig(mode="MODE_DYNAMIC", dynamic_threshold=0.5)
        assert config.mode == "MODE_DYNAMIC"
        assert config.dynamic_threshold == 0.5


class TestGoogleSearchRetrievalConfig:
    def test_defaults(self) -> None:
        config = GoogleSearchRetrievalConfig()
        assert config.dynamic_retrieval_config is None

    def test_with_config(self) -> None:
        config = GoogleSearchRetrievalConfig(
            dynamic_retrieval_config=DynamicRetrievalConfig(mode="MODE_DYNAMIC", dynamic_threshold=0.3),
        )
        assert config.dynamic_retrieval_config is not None
        assert config.dynamic_retrieval_config.mode == "MODE_DYNAMIC"
        assert config.dynamic_retrieval_config.dynamic_threshold == 0.3


class TestGCPVertexParams:
    def test_defaults(self) -> None:
        params = GCPVertexParams()
        assert params.safety_settings is None
        assert params.presence_penalty is None
        assert params.frequency_penalty is None
        assert params.seed is None
        assert params.labels is None
        assert params.thinking_config is None
        assert params.google_search is None
        assert params.google_search_retrieval is None
        assert params.code_execution is None
        assert params.task_type is None

    def test_all_fields(self) -> None:
        params = GCPVertexParams(
            safety_settings=[SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")],
            presence_penalty=0.5,
            frequency_penalty=0.3,
            seed=42,
            labels={"env": "test"},
            thinking_config={"thinking_budget": 1024},
            google_search=True,
            google_search_retrieval=GoogleSearchRetrievalConfig(),
            code_execution=True,
        )
        assert params.safety_settings is not None
        assert len(params.safety_settings) == 1
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == 0.3
        assert params.seed == 42
        assert params.labels == {"env": "test"}
        assert params.thinking_config == {"thinking_budget": 1024}
        assert params.google_search is True
        assert params.google_search_retrieval is not None
        assert params.code_execution is True

    def test_google_search_with_config(self) -> None:
        params = GCPVertexParams(
            google_search=GoogleSearchConfig(
                search_types=GoogleSearchTypes(web_search=True),
                exclude_domains=["example.com"],
            ),
        )
        assert isinstance(params.google_search, GoogleSearchConfig)
        assert params.google_search.search_types is not None
        assert params.google_search.search_types.web_search is True
        assert params.google_search.exclude_domains == ["example.com"]
