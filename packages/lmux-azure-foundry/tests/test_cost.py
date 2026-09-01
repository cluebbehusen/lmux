"""Tests for Azure AI Foundry pricing and cost calculation."""

import pytest

from lmux.types import Cost, Usage
from lmux_azure_foundry.cost import apply_cost_multiplier, calculate_azure_foundry_cost


class TestCalculateAzureFoundryCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_azure_foundry_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.00 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = calculate_azure_foundry_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_date_suffixed_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_azure_foundry_cost("gpt-4o-2024-11-20", usage)
        assert cost is not None
        base_cost = calculate_azure_foundry_cost("gpt-4o", usage)
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_with_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_azure_foundry_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 1.25 / 1_000_000)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=0)
        cost = calculate_azure_foundry_cost("text-embedding-3-small", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100 * 0.02 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_azure_foundry_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_prefix_matching_longest_first(self) -> None:
        """Verify that gpt-4o-mini-2024-07-18 matches gpt-4o-mini, not gpt-4o."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_azure_foundry_cost("gpt-4o-mini-2024-07-18", usage)
        assert cost is not None
        mini_cost = calculate_azure_foundry_cost("gpt-4o-mini", usage)
        assert mini_cost is not None
        assert cost.total_cost == pytest.approx(mini_cost.total_cost)

    def test_gpt_4o_2024_05_13_has_dedicated_pricing(self) -> None:
        """Azure prices the 05-13 snapshot at 5/15 with no cached-input rate, unlike the gpt-4o base."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_azure_foundry_cost("gpt-4o-2024-05-13", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((1000 - 200) * 5.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.00 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(0.0)
        base = calculate_azure_foundry_cost("gpt-4o", usage)
        assert base is not None
        assert cost.total_cost > base.total_cost

    def test_gpt5_model(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("gpt-5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1.25)
        assert cost.output_cost == pytest.approx(10.00)

    def test_deepseek_r1_model(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("deepseek-r1", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1.35)
        assert cost.output_cost == pytest.approx(5.40)

    def test_grok_model(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("grok-3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(3.00)
        assert cost.output_cost == pytest.approx(15.00)

    def test_llama_model(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("llama-4-maverick", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.25)
        assert cost.output_cost == pytest.approx(1.00)

    def test_mistral_model(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("mistral-large-3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.50)
        assert cost.output_cost == pytest.approx(1.50)

    @pytest.mark.parametrize("model", ["Cohere-command-a", "cohere-command-a", "COHERE-COMMAND-A"])
    def test_cohere_command_a_case_insensitive(self, model: str) -> None:
        # Lookup is case-insensitive, so the base id prices regardless of Azure's runtime casing.
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost(model, usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(2.50)
        assert cost.output_cost == pytest.approx(10.00)

    @pytest.mark.parametrize("model", ["Cohere-command-a-plus-05-2026", "Cohere-command-a-plus-06-2026"])
    def test_cohere_command_a_plus_version_agnostic(self, model: str) -> None:
        # Any dated Plus snapshot must keep the Plus rate, not fall through to the pricier base.
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost(model, usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.80)
        assert cost.output_cost == pytest.approx(3.20)

    def test_embed_v4(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        cost = calculate_azure_foundry_cost("embed-v-4-0", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.12)
        assert cost.output_cost == 0.0

    def test_mistral_large_3_capitalized(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("Mistral-Large-3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.50)
        assert cost.output_cost == pytest.approx(1.50)

    def test_mistral_medium_3_5(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("mistral-medium-3-5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1.50)
        assert cost.output_cost == pytest.approx(7.50)

    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate", "cache_rate"),
        [
            # Base is the Azure "K2.x ... glbl" meter; DZ deployments apply the multiplier on top.
            ("Kimi-K2.5", 0.60, 3.00, 0.10),
            ("Kimi-K2.6", 0.95, 4.00, 0.16),
            ("Kimi-K2.7-Code", 0.95, 4.00, 0.19),
        ],
    )
    def test_kimi_family(self, model: str, input_rate: float, output_rate: float, cache_rate: float) -> None:
        rates = calculate_azure_foundry_cost(model, Usage(input_tokens=1_000_000, output_tokens=1_000_000))
        cache = calculate_azure_foundry_cost(
            model, Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
        )
        assert rates is not None
        assert cache is not None
        assert rates.input_cost == pytest.approx(input_rate)
        assert rates.output_cost == pytest.approx(output_rate)
        assert cache.cache_read_cost == pytest.approx(cache_rate)

    def test_o1_pro_not_o1_prefix(self) -> None:
        """o1-pro must use its own 10x rate, not inherit the cheaper o1 prefix."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        pro = calculate_azure_foundry_cost("o1-pro", usage)
        base = calculate_azure_foundry_cost("o1", usage)
        assert pro is not None
        assert base is not None
        assert pro.input_cost == pytest.approx(150.00)
        assert pro.output_cost == pytest.approx(600.00)
        assert pro.input_cost > base.input_cost
        cache = calculate_azure_foundry_cost(
            "o1-pro", Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
        )
        assert cache is not None
        assert cache.cache_read_cost == pytest.approx(75.00)

    def test_gpt_5_2_pro_not_gpt_5_2_prefix(self) -> None:
        """gpt-5.2-pro uses its own ~12x rate with no cache meter, not the gpt-5.2 prefix."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("gpt-5.2-pro", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(21.00)
        assert cost.output_cost == pytest.approx(168.00)

    def test_gpt_chat_latest(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("gpt-chat-latest", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(5.00)
        assert cost.output_cost == pytest.approx(30.00)

    def test_grok_4_1_fast_dash_ids(self) -> None:
        """Dash-form grok-4-1-fast-* ids match grok-4-1-fast (0.20/0.50), not grok-4 (3/15)."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        for model in ("grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning"):
            cost = calculate_azure_foundry_cost(model, usage)
            assert cost is not None
            assert cost.input_cost == pytest.approx(0.20)
            assert cost.output_cost == pytest.approx(0.50)

    def test_grok_4_20_preview_unpriced_returns_none(self) -> None:
        """grok-4-20-* Preview ids have no published rate and must return None, not fall through to grok-4."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        assert calculate_azure_foundry_cost("grok-4-20-reasoning", usage) is None
        assert calculate_azure_foundry_cost("grok-4-20-non-reasoning", usage) is None

    def test_deepseek_v4_flash_0731_does_not_inherit_the_base_flash_rate(self) -> None:
        """The 0731 snapshot is priced above the base model, so it needs its own key."""
        usage = Usage(input_tokens=2_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)
        snapshot = calculate_azure_foundry_cost("DeepSeek-V4-Flash-0731", usage)
        assert snapshot is not None
        assert snapshot.input_cost == pytest.approx(0.44)
        assert snapshot.output_cost == pytest.approx(1.32)
        assert snapshot.cache_read_cost == pytest.approx(0.014)

    def test_deepseek_v4_cache_reads(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
        flash = calculate_azure_foundry_cost("deepseek-v4-flash", usage)
        pro = calculate_azure_foundry_cost("deepseek-v4-pro", usage)
        assert flash is not None
        assert pro is not None
        assert flash.cache_read_cost == pytest.approx(0.028)
        assert pro.cache_read_cost == pytest.approx(0.145)

    @pytest.mark.parametrize(
        ("model", "base_rates", "hi_rates"),
        [
            ("gpt-5.6-sol", (5.00, 30.00, 0.50, 6.25), (10.00, 45.00, 1.00, 12.50)),
            ("gpt-5.6-terra", (2.00, 12.00, 0.20, 2.50), (4.00, 18.00, 0.40, 5.00)),
            ("gpt-5.6-luna", (0.20, 1.20, 0.02, 0.25), (0.40, 1.80, 0.04, 0.50)),
        ],
    )
    def test_gpt_5_6_family_rates_including_cache_write(
        self,
        model: str,
        base_rates: tuple[float, float, float, float],
        hi_rates: tuple[float, float, float, float],
    ) -> None:
        """gpt-5.6-* bill input/output/cache-read/cache-write on both tiers, cache write at 1.25x input."""
        in_base, out_base, cr_base, cw_base = base_rates
        in_hi, out_hi, cr_hi, cw_hi = hi_rates
        base = calculate_azure_foundry_cost(
            model, Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100, cache_creation_tokens=200)
        )
        assert base is not None
        assert base.input_cost == pytest.approx((1000 - 100 - 200) * in_base / 1_000_000)
        assert base.output_cost == pytest.approx(500 * out_base / 1_000_000)
        assert base.cache_read_cost == pytest.approx(100 * cr_base / 1_000_000)
        assert base.cache_creation_cost == pytest.approx(200 * cw_base / 1_000_000)
        hi = calculate_azure_foundry_cost(model, Usage(input_tokens=300_000, output_tokens=1000))
        assert hi is not None
        assert hi.input_cost == pytest.approx(300_000 * in_hi / 1_000_000)
        assert hi.output_cost == pytest.approx(1000 * out_hi / 1_000_000)
        cache_hi = calculate_azure_foundry_cost(
            model,
            Usage(input_tokens=300_000, output_tokens=0, cache_read_tokens=200_000, cache_creation_tokens=100_000),
        )
        assert cache_hi is not None
        assert cache_hi.cache_read_cost == pytest.approx(200_000 * cr_hi / 1_000_000)
        assert cache_hi.cache_creation_cost == pytest.approx(100_000 * cw_hi / 1_000_000)

    def test_gpt_5_6_bare_alias_matches_sol(self) -> None:
        """The bare gpt-5.6 alias resolves to Sol rates, not the gpt-5 prefix (1.25/10)."""
        usage = Usage(input_tokens=100_000, output_tokens=100_000)
        bare = calculate_azure_foundry_cost("gpt-5.6", usage)
        sol = calculate_azure_foundry_cost("gpt-5.6-sol", usage)
        assert bare is not None
        assert sol is not None
        assert bare.input_cost == pytest.approx(100_000 * 5.00 / 1_000_000)
        assert bare.total_cost == pytest.approx(sol.total_cost)

    def test_grok_4_2_corrected_pricing(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_azure_foundry_cost("grok-4.2", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1.25)
        assert cost.output_cost == pytest.approx(2.50)

    def test_grok_4_3_model(self) -> None:
        usage = Usage(input_tokens=100_000, output_tokens=100_000, cache_read_tokens=10_000)
        cost = calculate_azure_foundry_cost("grok-4.3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((100_000 - 10_000) * 1.25 / 1_000_000)
        assert cost.output_cost == pytest.approx(100_000 * 2.50 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(10_000 * 0.20 / 1_000_000)

    def test_grok_4_3_long_context_tier(self) -> None:
        """Every grok-4.3 rate doubles above 200K prompt tokens."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=100_000)
        cost = calculate_azure_foundry_cost("grok-4.3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((1_000_000 - 100_000) * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(1_000_000 * 5.00 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(100_000 * 0.40 / 1_000_000)

    def test_deepseek_v4_models(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        pro = calculate_azure_foundry_cost("deepseek-v4-pro", usage)
        flash = calculate_azure_foundry_cost("deepseek-v4-flash", usage)
        assert pro is not None
        assert flash is not None
        assert pro.input_cost == pytest.approx(1.74)
        assert pro.output_cost == pytest.approx(3.48)
        assert flash.input_cost == pytest.approx(0.19)
        assert flash.output_cost == pytest.approx(0.51)

    def test_gpt_5_5_base_tier(self) -> None:
        usage = Usage(input_tokens=100_000, output_tokens=100_000)
        cost = calculate_azure_foundry_cost("gpt-5.5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100_000 * 5.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(100_000 * 30.00 / 1_000_000)

    def test_gpt_5_5_long_context_tier(self) -> None:
        usage = Usage(input_tokens=300_000, output_tokens=1000)
        cost = calculate_azure_foundry_cost("gpt-5.5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(300_000 * 10.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 45.00 / 1_000_000)

    def test_phi_4_multimodal_does_not_fall_back_to_base(self) -> None:
        """Phi-4-multimodal-instruct must price at the multimodal (text/image) rate, not the broad Phi-4 key."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        multimodal = calculate_azure_foundry_cost("Phi-4-multimodal-instruct", usage)
        base = calculate_azure_foundry_cost("Phi-4", usage)
        assert multimodal is not None
        assert base is not None
        assert multimodal.input_cost == pytest.approx(0.08)
        assert multimodal.output_cost == pytest.approx(0.32)
        assert base.input_cost == pytest.approx(0.125)
        assert base.output_cost == pytest.approx(0.50)


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

    def test_none_cache_fields_stay_none(self) -> None:
        cost = Cost(input_cost=1.0, output_cost=2.0, total_cost=3.0)
        result = apply_cost_multiplier(cost, 2.0)
        assert result.input_cost == pytest.approx(2.0)
        assert result.output_cost == pytest.approx(4.0)
        assert result.total_cost == pytest.approx(6.0)
        assert result.cache_read_cost is None
        assert result.cache_creation_cost is None
