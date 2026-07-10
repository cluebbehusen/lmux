"""Tests for the pure weighted-selection helpers."""

import pytest

from lmux import InvalidRequestError
from lmux_load_balancer._selection import ordered_candidates, point_for, validate_group


class TestValidateGroup:
    def test_empty_group_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="no endpoints"):
            validate_group("m", {})

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="invalid weight"):
            validate_group("m", {"a/x": -1.0})

    def test_non_finite_weight_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="invalid weight"):
            validate_group("m", {"a/x": float("inf")})

    def test_zero_sum_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="positive value"):
            validate_group("m", {"a/x": 0.0, "b/x": 0.0})

    def test_valid_group_passes(self) -> None:
        validate_group("m", {"a/x": 1.0, "b/x": 0.0})  # a disabled endpoint is allowed


class TestPointFor:
    def test_keyless_is_in_range(self) -> None:
        for _ in range(100):
            assert 0.0 <= point_for(None) < 1.0

    def test_sticky_is_deterministic(self) -> None:
        assert point_for("chat-1") == point_for("chat-1")
        assert 0.0 <= point_for("chat-1") < 1.0

    def test_sticky_pinned_value(self) -> None:
        # Pinned SHA-256-derived value: guards against a regression to builtin hash().
        assert point_for("lb-sticky") == 0.9818856505788531


class TestOrderedCandidates:
    def test_orders_by_cdf_at_low_point(self) -> None:
        assert ordered_candidates({"a/x": 1.0, "b/x": 1.0}, 0.0) == ["a/x", "b/x"]

    def test_last_bucket_is_catch_all(self) -> None:
        # A point at the top of the distribution falls through to the last bucket.
        assert ordered_candidates({"a/x": 1.0, "b/x": 1.0}, 0.999999) == ["b/x", "a/x"]

    def test_zero_weight_endpoints_excluded(self) -> None:
        assert ordered_candidates({"a/x": 0.0, "b/x": 1.0}, 0.0) == ["b/x"]

    def test_single_endpoint(self) -> None:
        assert ordered_candidates({"a/x": 1.0}, 0.5) == ["a/x"]
