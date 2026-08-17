"""Tests for Google cost calculation."""

from datetime import date

import pytest

from lmux.types import Cost, Usage
from lmux_google.cost import apply_cost_multiplier, calculate_google_cost, has_vertex_non_global_premium


class TestCalculateGoogleCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.15 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.60 / 1_000_000)
        assert cost.total_cost == cost.input_cost + cost.output_cost

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        assert calculate_google_cost("totally-unknown-model", usage) is None

    def test_prefix_match_with_date_suffix(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.0-flash-001", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.15 / 1_000_000)

    def test_tiered_pricing_below_threshold(self) -> None:
        usage = Usage(input_tokens=100_000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.5-pro", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100_000 * 1.25 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.00 / 1_000_000)

    def test_tiered_pricing_above_threshold(self) -> None:
        usage = Usage(input_tokens=300_000, output_tokens=1000)
        cost = calculate_google_cost("gemini-2.5-pro", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(300_000 * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 15.00 / 1_000_000)

    def test_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_google_cost("gemini-2.5-pro", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.125 / 1_000_000)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0)
        cost = calculate_google_cost("text-embedding-005", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.10 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_gemini_embedding_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0)
        cost = calculate_google_cost("gemini-embedding-001", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.15 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_computer_use_base_tier_has_no_cache_pricing(self) -> None:
        """The dedicated computer-use key prices the base tier and has no cache rate.

        gemini-2.5-pro shares the same base/>200K token rates, so the no-cache
        signal is what distinguishes the dedicated key: if it were dropped and
        prefix-matched to gemini-2.5-pro (cache 0.125/M), cache_read_cost would
        be non-zero.
        """
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_google_cost("gemini-2.5-computer-use-preview-10-2025", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((1000 - 200) * 1.25 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.00 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(0.0)

    def test_computer_use_long_context_tier(self) -> None:
        """gemini-2.5-computer-use-preview-10-2025 uses the >200K tier above 200K input tokens."""
        usage = Usage(input_tokens=250_000, output_tokens=1000)
        cost = calculate_google_cost("gemini-2.5-computer-use-preview-10-2025", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(250_000 * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 15.00 / 1_000_000)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_google_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_longest_prefix_match(self) -> None:
        """gemini-2.0-flash-lite should match before gemini-2.0-flash."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_lite = calculate_google_cost("gemini-2.0-flash-lite-001", usage)
        cost_flash = calculate_google_cost("gemini-2.0-flash-001", usage)
        assert cost_lite is not None
        assert cost_flash is not None
        assert cost_lite.input_cost < cost_flash.input_cost

    def test_case_insensitive_lookup(self) -> None:
        """A capitalized model id resolves identically to its lowercase form."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        upper = calculate_google_cost("GEMINI-2.5-PRO", usage)
        lower = calculate_google_cost("gemini-2.5-pro", usage)
        assert upper is not None
        assert lower is not None
        assert upper.total_cost == pytest.approx(lower.total_cost)

    def test_partner_model_not_priced(self) -> None:
        """Partner models (Claude, Mistral, ...) are not servable by this provider, so they have no pricing."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        assert calculate_google_cost("claude-sonnet-4-6", usage) is None
        assert calculate_google_cost("mistral-medium-3", usage) is None

    def test_computer_use_real_id_priced_at_base_rate(self) -> None:
        """The real computer-use id resolves to the dedicated (no-cache) rate.

        The old mis-named key gemini-2.5-pro-computer-use-preview was never a real runtime id;
        it now harmlessly prefix-matches gemini-2.5-pro, so it is not asserted here.
        """
        usage = Usage(input_tokens=1000, output_tokens=500)
        real = calculate_google_cost("gemini-2.5-computer-use-preview-10-2025", usage)
        assert real is not None
        assert real.input_cost == pytest.approx(1000 * 1.25 / 1_000_000)
        assert real.output_cost == pytest.approx(500 * 10.00 / 1_000_000)

    @pytest.mark.parametrize(
        ("ga", "preview", "input_rate"),
        [
            ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", 0.25),
            ("gemini-embedding-2", "gemini-embedding-2-preview", 0.20),
        ],
    )
    def test_ga_ids_match_shutdown_previews_and_pin_rate(self, ga: str, preview: str, input_rate: float) -> None:
        """GA ids resolve to the same rates as their shut-down -preview aliases, pinned to reality."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        ga_cost = calculate_google_cost(ga, usage)
        preview_cost = calculate_google_cost(preview, usage)
        assert ga_cost is not None
        assert preview_cost is not None
        assert ga_cost.input_cost == pytest.approx(1000 * input_rate / 1_000_000)
        assert ga_cost.total_cost == pytest.approx(preview_cost.total_cost)

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image",
            # -preview and dated variants must also be caught (prefix sentinel), or they would
            # fall through to a text-priced base (e.g. gemini-2.5-flash) and mis-report image gen.
            "gemini-2.5-flash-image-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
            "gemini-3-pro-image-001",
        ],
    )
    def test_image_output_models_unpriced(self, model: str) -> None:
        """Image-output models (and their dated/-preview variants) return None: image output is
        billed far above the text rate, which a single output rate would underprice ~10-20x."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        assert calculate_google_cost(model, usage) is None
        assert calculate_google_cost(model.upper(), usage) is None

    def test_robotics_er_priced(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_google_cost("gemini-robotics-er-1.6-preview", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 1.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 5.00 / 1_000_000)

    def test_gemini_3_1_flash_lite_cache_rate(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0, cache_read_tokens=500)
        cost = calculate_google_cost("gemini-3.1-flash-lite", usage)
        assert cost is not None
        assert cost.cache_read_cost == pytest.approx(500 * 0.025 / 1_000_000)

    @pytest.mark.parametrize(
        "model",
        ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite-preview-06-2026", "GEMINI-3.5-Flash-Lite"],
    )
    def test_gemini_3_5_flash_lite_does_not_fall_through_to_flash(self, model: str) -> None:
        """Flash-Lite must not prefix-match gemini-3.5-flash, which bills 5x its input rate."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)
        cost = calculate_google_cost(model, usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.0)  # every input token is a cache read
        assert cost.output_cost == pytest.approx(2.50)
        assert cost.cache_read_cost == pytest.approx(0.03)

    @pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.7-flash"])
    def test_gemini_3_flash_introductory_pricing(self, model: str) -> None:
        """Through 2026-12-31 both Flash models bill the introductory rate."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=100_000)
        cost = calculate_google_cost(model, usage, as_of=date(2026, 8, 16))
        assert cost is not None
        assert cost.input_cost == pytest.approx((1_000_000 - 100_000) * 0.75 / 1_000_000)
        assert cost.output_cost == pytest.approx(3.75)
        assert cost.cache_read_cost == pytest.approx(100_000 * 0.075 / 1_000_000)

    @pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.7-flash"])
    def test_gemini_3_flash_standard_pricing_from_2027(self, model: str) -> None:
        """From 2027-01-01 the introductory window ends and every rate doubles."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=100_000)
        cost = calculate_google_cost(model, usage, as_of=date(2027, 1, 1))
        assert cost is not None
        assert cost.input_cost == pytest.approx((1_000_000 - 100_000) * 1.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(7.50)
        assert cost.cache_read_cost == pytest.approx(100_000 * 0.15 / 1_000_000)

    @pytest.mark.parametrize(
        ("model", "rate"),
        [("gemini-2.0-flash", 0.0375), ("gemini-2.0-flash-lite", 0.01875)],
    )
    def test_gemini_2_0_cache_reads_are_billed(self, model: str, rate: float) -> None:
        """An unset cache-read rate would bill these tokens at zero rather than the published rate."""
        usage = Usage(input_tokens=100_000, output_tokens=0, cache_read_tokens=100_000)
        cost = calculate_google_cost(model, usage)
        assert cost is not None
        assert cost.cache_read_cost == pytest.approx(100_000 * rate / 1_000_000)
        assert cost.total_cost > 0.0


class TestHasVertexNonGlobalPremium:
    @pytest.mark.parametrize(
        "model",
        ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash-002"],
    )
    def test_models_with_a_published_non_global_rate(self, model: str) -> None:
        assert has_vertex_non_global_premium(model) is True

    @pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.7-flash"])
    def test_ga_models_without_a_published_non_global_rate(self, model: str) -> None:
        """Being GA is not the test: Vertex publishes no Non-global row for these two."""
        assert has_vertex_non_global_premium(model) is False

    def test_preview_id_is_exempt_despite_matching_a_ga_prefix(self) -> None:
        """Longest-prefix wins: the -preview id must not inherit its GA sibling's premium."""
        assert has_vertex_non_global_premium("gemini-3.1-flash-lite-preview") is False
        assert has_vertex_non_global_premium("gemini-3.1-flash-lite") is True

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "text-embedding-005",
            "gemini-embedding-001",
            "totally-unknown-model",
        ],
    )
    def test_other_models_are_exempt(self, model: str) -> None:
        assert has_vertex_non_global_premium(model) is False


class TestApplyCostMultiplier:
    def test_applies_multiplier_to_all_fields(self) -> None:
        cost = Cost(
            input_cost=1.0,
            output_cost=2.0,
            total_cost=3.0,
            cache_read_cost=0.5,
            cache_creation_cost=0.25,
        )
        result = apply_cost_multiplier(cost, 1.1)
        assert result.input_cost == pytest.approx(1.1)
        assert result.output_cost == pytest.approx(2.2)
        assert result.total_cost == pytest.approx(3.3)
        assert result.cache_read_cost == pytest.approx(0.55)
        assert result.cache_creation_cost == pytest.approx(0.275)

    def test_preserves_none_cache_costs(self) -> None:
        cost = Cost(input_cost=1.0, output_cost=2.0, total_cost=3.0)
        result = apply_cost_multiplier(cost, 2.0)
        assert result.cache_read_cost is None
        assert result.cache_creation_cost is None
