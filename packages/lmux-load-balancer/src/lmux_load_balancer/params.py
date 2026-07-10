"""Per-call parameters for the load-balancer provider."""

from typing import Literal

from lmux import BaseProviderParams


class LoadBalancerParams(BaseProviderParams):
    """Routing parameters for :class:`~lmux_load_balancer.provider.LoadBalancerProvider`.

    ``sticky_key`` pins a caller-defined identity (e.g. a conversation id) to a
    single endpoint so its provider-side state (such as a prompt cache) stays warm;
    without one, each call is distributed by weight independently.

    ``failover`` controls whether a failed endpoint falls through to the next
    candidate: ``"always"`` (default), ``"never"``, or ``"unless_sticky"`` (fall
    through only for keyless calls, keeping sticky calls pinned to their endpoint).
    """

    sticky_key: str | None = None
    failover: Literal["never", "always", "unless_sticky"] = "always"
