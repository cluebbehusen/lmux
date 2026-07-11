"""Response metadata for the load-balancer provider."""

from lmux import BaseProviderMetadata


class LoadBalancerMetadata(BaseProviderMetadata):
    """Which endpoint served a load-balanced request, and whether it failed over.

    ``primary`` is the endpoint the routing key selected; ``served`` is the one that
    actually produced the response; ``attempted`` lists the endpoints tried in order
    (a length greater than one means at least one failover occurred).
    """

    primary: str
    served: str
    attempted: list[str]
