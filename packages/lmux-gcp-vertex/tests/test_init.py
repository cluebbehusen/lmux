"""Tests for the lmux-gcp-vertex compatibility shim."""

import importlib

import pytest

import lmux_gcp_vertex
import lmux_gcp_vertex.auth
import lmux_gcp_vertex.cost
import lmux_gcp_vertex.params
import lmux_gcp_vertex.provider
import lmux_google
import lmux_google.provider


class TestShim:
    def test_emits_deprecation_warning_on_import(self) -> None:
        with pytest.warns(DeprecationWarning, match="renamed to lmux-google"):
            _ = importlib.reload(lmux_gcp_vertex)

    def test_old_names_alias_new_objects(self) -> None:
        assert lmux_gcp_vertex.GCPVertexProvider is lmux_google.GoogleProvider
        assert lmux_gcp_vertex.GCPVertexParams is lmux_google.GoogleParams
        assert lmux_gcp_vertex.GCPVertexADCAuthProvider is lmux_google.GoogleADCAuthProvider
        assert lmux_gcp_vertex.GCPVertexAPIKeyAuthProvider is lmux_google.GoogleAPIKeyAuthProvider
        assert lmux_gcp_vertex.GCPVertexServiceAccountAuthProvider is lmux_google.GoogleServiceAccountAuthProvider
        assert lmux_gcp_vertex.calculate_gcp_vertex_cost is lmux_google.calculate_google_cost

    def test_unrenamed_names_reexported(self) -> None:
        assert lmux_gcp_vertex.DynamicRetrievalConfig is lmux_google.DynamicRetrievalConfig
        assert lmux_gcp_vertex.GoogleSearchConfig is lmux_google.GoogleSearchConfig
        assert lmux_gcp_vertex.GoogleSearchRetrievalConfig is lmux_google.GoogleSearchRetrievalConfig
        assert lmux_gcp_vertex.GoogleSearchTypes is lmux_google.GoogleSearchTypes
        assert lmux_gcp_vertex.SafetySetting is lmux_google.SafetySetting
        assert lmux_gcp_vertex.preload is lmux_google.preload

    def test_submodule_imports_forward(self) -> None:
        """0.6.x submodule import paths keep working."""
        assert lmux_gcp_vertex.provider.GCPVertexProvider is lmux_google.GoogleProvider
        assert lmux_gcp_vertex.provider.GCPVertexAuth is lmux_google.provider.GoogleAuth
        assert lmux_gcp_vertex.provider.PROVIDER_NAME == "google"
        assert lmux_gcp_vertex.auth.GCPVertexADCAuthProvider is lmux_google.GoogleADCAuthProvider
        assert lmux_gcp_vertex.auth.GCPVertexAPIKeyAuthProvider is lmux_google.GoogleAPIKeyAuthProvider
        assert lmux_gcp_vertex.auth.GCPVertexServiceAccountAuthProvider is lmux_google.GoogleServiceAccountAuthProvider
        assert lmux_gcp_vertex.params.GCPVertexParams is lmux_google.GoogleParams
        assert lmux_gcp_vertex.params.SafetySetting is lmux_google.SafetySetting
        assert lmux_gcp_vertex.cost.calculate_gcp_vertex_cost is lmux_google.calculate_google_cost

    def test_all_matches_old_public_api(self) -> None:
        assert lmux_gcp_vertex.__all__ == [
            "DynamicRetrievalConfig",
            "GCPVertexADCAuthProvider",
            "GCPVertexAPIKeyAuthProvider",
            "GCPVertexParams",
            "GCPVertexProvider",
            "GCPVertexServiceAccountAuthProvider",
            "GoogleSearchConfig",
            "GoogleSearchRetrievalConfig",
            "GoogleSearchTypes",
            "SafetySetting",
            "calculate_gcp_vertex_cost",
            "preload",
        ]
