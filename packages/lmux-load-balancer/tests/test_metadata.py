"""Tests for LoadBalancerMetadata."""

from lmux import BaseProviderMetadata
from lmux_load_balancer import LoadBalancerMetadata


class TestLoadBalancerMetadata:
    def test_fields(self) -> None:
        meta = LoadBalancerMetadata(primary="a/x", served="b/x", attempted=["a/x", "b/x"])
        assert meta.primary == "a/x"
        assert meta.served == "b/x"
        assert meta.attempted == ["a/x", "b/x"]

    def test_is_base_provider_metadata(self) -> None:
        meta = LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])
        assert isinstance(meta, BaseProviderMetadata)
