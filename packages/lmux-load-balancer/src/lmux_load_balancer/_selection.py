"""Deterministic weighted selection for the load balancer.

Pure functions with no cross-request state: a stable routing key hashes to a fixed
point in ``[0, 1)``, and the weighted cumulative distribution turns that point into a
deterministic, per-key ordering of endpoints. Weights hold only in aggregate across
many distinct keys, not per call.
"""

import hashlib
import math
import random
from collections.abc import Mapping

from lmux import InvalidRequestError


def validate_group(logical_model: str, weights: Mapping[str, float]) -> None:
    """Raise :class:`~lmux.InvalidRequestError` if an endpoint group is unusable."""
    if not weights:
        msg = f"Load-balancer group {logical_model!r} has no endpoints"
        raise InvalidRequestError(msg)
    for endpoint, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            msg = f"Load-balancer group {logical_model!r} endpoint {endpoint!r} has invalid weight {weight}"
            raise InvalidRequestError(msg)
    if sum(weights.values()) <= 0:
        msg = f"Load-balancer group {logical_model!r} weights must sum to a positive value"
        raise InvalidRequestError(msg)


def point_for(sticky_key: str | None) -> float:
    """A point in ``[0, 1)``: deterministic from a sticky key, else random per call.

    Uses SHA-256 (stable across processes) so the same key routes identically on every
    host and invocation; builtin ``hash()`` is salted per process and must not be used.
    """
    if sticky_key is None:
        # Non-cryptographic: this only spreads keyless load across endpoints.
        return random.random()  # noqa: S311
    digest = hashlib.sha256(sticky_key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


def ordered_candidates(weights: Mapping[str, float], point: float) -> list[str]:
    """Deterministic endpoint order for ``point``: primary first, then failover order.

    Each position is the weighted-CDF bucket ``point`` falls into over the endpoints not
    yet chosen (renormalized, same point), so the whole chain is deterministic per key.
    A weight of ``0`` disables an endpoint: it is excluded from selection and failover.
    Assumes ``weights`` has passed :func:`validate_group`.
    """
    remaining = {endpoint: weight for endpoint, weight in weights.items() if weight > 0}
    order: list[str] = []
    while remaining:
        choice = _pick(remaining, point)
        order.append(choice)
        del remaining[choice]
    return order


def _pick(weights: Mapping[str, float], point: float) -> str:
    """Pick the weighted-CDF bucket ``point`` falls into. ``weights`` is non-empty, all > 0."""
    endpoints = list(weights)
    total = sum(weights.values())
    cumulative = 0.0
    for endpoint in endpoints[:-1]:
        cumulative += weights[endpoint] / total
        if point < cumulative:
            return endpoint
    # The last bucket is the catch-all, which also guards float drift and point ~= 1.0.
    return endpoints[-1]
