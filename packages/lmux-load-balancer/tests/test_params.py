"""Tests for LoadBalancerParams."""

from lmux import BaseProviderParams
from lmux_load_balancer import LoadBalancerParams


class TestLoadBalancerParams:
    def test_defaults(self) -> None:
        params = LoadBalancerParams()
        assert params.sticky_key is None
        assert params.failover == "always"

    def test_is_base_provider_params(self) -> None:
        assert isinstance(LoadBalancerParams(), BaseProviderParams)

    def test_explicit_values(self) -> None:
        params = LoadBalancerParams(sticky_key="chat-1", failover="never")
        assert params.sticky_key == "chat-1"
        assert params.failover == "never"
