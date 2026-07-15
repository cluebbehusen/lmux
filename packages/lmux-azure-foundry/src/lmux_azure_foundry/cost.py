"""Azure AI Foundry pricing data and cost calculation.

Prices are Global Standard (pay-as-you-go) pricing.  Data Zone and Regional
deployments apply a multiplier on top of these base rates.

Use ``register_pricing()`` on ``AzureFoundryProvider`` for provisioned
deployments or models not listed here.

Pricing source: OpenAI (AOAI) models at https://azure.microsoft.com/en-us/pricing/details/azure-openai/.
Models sold directly by Azure (DeepSeek, Grok, Llama, Mistral, Cohere, Phi) are on the per-vendor Foundry
Models pages, e.g. https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/microsoft/ for Phi;
swap the trailing path segment for the vendor (/deepseek, /grok, /llama, /mistral-ai, /cohere, /kimi).
"""

from lmux.cost import (
    ModelPricing,
    PricingTier,
    build_pricing_index,
    calculate_cost,
    per_million_tokens,
    resolve_pricing,
)
from lmux.types import Cost, Usage

# MARK: Deployment-type multipliers

DATA_ZONE_MULTIPLIER = 1.1
"""Commercial US/EU Data Zone Standard deployments are 1.1x global pricing.

This holds uniformly across input, output, and cache for every model, and the
US and EU Data Zone prices are identical. Non-commercial data zones are not
modeled and run higher — the Asia-Pacific data zone is ~1.2x and the US
Government sovereign cloud is ~1.375x; use ``register_pricing()`` for those.
"""

REGIONAL_MULTIPLIER = 1.1
"""Regional deployments are approximately 1.1x global pricing.

Note: actual regional pricing varies by model (1.1x-1.375x).  This constant
uses the most common multiplier; for exact per-model regional rates, use
``register_pricing()`` to override individual models.
"""

# MARK: Global Standard pricing (base rates)

_PRICING: dict[str, ModelPricing] = {
    # --- OpenAI: GPT-5 family ---
    "gpt-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(2.00),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    "gpt-5-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.05),
                output_cost_per_token=per_million_tokens(0.40),
                cache_read_cost_per_token=per_million_tokens(0.005),
            )
        ],
    ),
    # --- OpenAI: GPT-5.6 family (sol/terra/luna) ---
    # Azure had not published gpt-5.6 retail meters at the time of writing (GA on Azure but
    # unpriced), so these mirror OpenAI's Global Standard rates, which Azure AOAI meters match
    # for every other GPT-5.x model. Cache-write is intentionally omitted: Azure does not meter
    # cache writes for any AOAI model (unlike OpenAI, which bills gpt-5.6+ cache writes). Replace
    # with the -glbl meters once Azure publishes them. The bare "gpt-5.6" alias routes to Sol.
    "gpt-5.6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.6-sol": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.6-terra": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.6-luna": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(6.00),
                cache_read_cost_per_token=per_million_tokens(0.10),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(9.00),
                cache_read_cost_per_token=per_million_tokens(0.20),
                min_input_tokens=272_000,
            ),
        ],
    ),
    # --- OpenAI: GPT-5.5 family ---
    "gpt-5.5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    # --- OpenAI: GPT-5.4 family ---
    "gpt-5.4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.4-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(30.00),
                output_cost_per_token=per_million_tokens(180.00),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(60.00),
                output_cost_per_token=per_million_tokens(270.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.4-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.75),
                output_cost_per_token=per_million_tokens(4.50),
                cache_read_cost_per_token=per_million_tokens(0.075),
            )
        ],
    ),
    "gpt-5.4-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(1.25),
                cache_read_cost_per_token=per_million_tokens(0.02),
            )
        ],
    ),
    # --- OpenAI: GPT-5.3 family ---
    "gpt-5.3-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.3-chat": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    # --- OpenAI: GPT-4.5 ---
    # Deprecated: gpt-4.5-preview was retired on Azure 2025-07-14 (replaced by gpt-4.1).
    # Kept for historical cost lookups.
    "gpt-4.5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(75.00),
                output_cost_per_token=per_million_tokens(150.00),
                cache_read_cost_per_token=per_million_tokens(37.50),
            )
        ],
    ),
    "gpt-5.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    # gpt-5.2-pro is ~12x gpt-5.2 on Azure and has no cached-input meter; explicit key
    # stops it inheriting the far cheaper "gpt-5.2" prefix (and its spurious cache rate).
    "gpt-5.2-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(21.00),
                output_cost_per_token=per_million_tokens(168.00),
            )
        ],
    ),
    "gpt-5.2-chat": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.2-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5.1-chat": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5.1-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5.1-codex-max": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5.1-codex-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(2.00),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    "gpt-5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(120.00),
            )
        ],
    ),
    # Preview chat alias (replacement for the retired gpt-5.1/5.2/5.3-chat models).
    "gpt-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # --- OpenAI: GPT-4.1 family ---
    "gpt-4.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "gpt-4.1-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.40),
                output_cost_per_token=per_million_tokens(1.60),
                cache_read_cost_per_token=per_million_tokens(0.10),
            )
        ],
    ),
    "gpt-4.1-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=per_million_tokens(0.40),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    # --- OpenAI: GPT-4o family ---
    "gpt-4o-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
                cache_read_cost_per_token=per_million_tokens(0.075),
            )
        ],
    ),
    "gpt-4o": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(1.25),
            )
        ],
    ),
    # --- OpenAI: Reasoning models ---
    "o1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(60.00),
                cache_read_cost_per_token=per_million_tokens(7.50),
            )
        ],
    ),
    # o1-pro is 10x o1 on Azure; explicit key stops it inheriting the cheaper "o1" prefix.
    "o1-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(150.00),
                output_cost_per_token=per_million_tokens(600.00),
                cache_read_cost_per_token=per_million_tokens(75.00),
            )
        ],
    ),
    "o3-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(20.00),
                output_cost_per_token=per_million_tokens(80.00),
            )
        ],
    ),
    "o3-deep-research": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(40.00),
                cache_read_cost_per_token=per_million_tokens(2.50),
            )
        ],
    ),
    "o3-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.55),
            )
        ],
    ),
    "o3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "o4-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.275),
            )
        ],
    ),
    "o1-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.55),
            )
        ],
    ),
    "computer-use-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(12.00),
            )
        ],
    ),
    "codex-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.50),
                output_cost_per_token=per_million_tokens(6.00),
                cache_read_cost_per_token=per_million_tokens(0.375),
            )
        ],
    ),
    # --- OpenAI: Embedding models ---
    # Global Standard rates (OpenAI parity); the +10% data-zone/regional premium is
    # applied via DATA_ZONE_MULTIPLIER / REGIONAL_MULTIPLIER, not baked into the base.
    "text-embedding-3-small": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.02), output_cost_per_token=0.0)]
    ),
    "text-embedding-3-large": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.13), output_cost_per_token=0.0)]
    ),
    "text-embedding-ada-002": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.10), output_cost_per_token=0.0)]
    ),
    # --- DeepSeek ---
    "deepseek-r1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.35),
                output_cost_per_token=per_million_tokens(5.40),
            )
        ],
    ),
    "deepseek-v3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.14),
                output_cost_per_token=per_million_tokens(4.56),
            )
        ],
    ),
    "deepseek-v3.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.23),
                output_cost_per_token=per_million_tokens(4.94),
            )
        ],
    ),
    "deepseek-v3.2-sp": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.58),
                output_cost_per_token=per_million_tokens(1.68),
            )
        ],
    ),
    "deepseek-v3.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.58),
                output_cost_per_token=per_million_tokens(1.68),
            )
        ],
    ),
    "deepseek-v4-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.74),
                output_cost_per_token=per_million_tokens(3.48),
                cache_read_cost_per_token=per_million_tokens(0.145),
            )
        ],
    ),
    "deepseek-v4-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.19),
                output_cost_per_token=per_million_tokens(0.51),
                cache_read_cost_per_token=per_million_tokens(0.028),
            )
        ],
    ),
    # --- xAI (Grok) ---
    "grok-3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    "grok-3-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.27),
            )
        ],
    ),
    "grok-4.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(2.50),
            )
        ],
    ),
    "grok-4.3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(2.50),
            )
        ],
    ),
    "grok-4-fast": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # "Grok 4.1 Fast" GA (replaces the retired grok-4-fast). Runtime ids are the dash form
    # grok-4-1-fast-reasoning / grok-4-1-fast-non-reasoning, which the dot-form "grok-4.1" key
    # never matches; without this key they fall through to "grok-4" (3/15) and overcharge ~30x.
    "grok-4-1-fast": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "grok-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    "grok-code-fast-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(1.50),
            )
        ],
    ),
    "grok-4.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # --- Meta (Llama) ---
    "llama-3.3-70b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.71),
                output_cost_per_token=per_million_tokens(0.71),
            )
        ],
    ),
    "llama-4-scout": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.16),
                output_cost_per_token=per_million_tokens(0.64),
            )
        ],
    ),
    "llama-4-maverick": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.00),
            )
        ],
    ),
    # --- Mistral ---
    "mistral-large": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.50),
                output_cost_per_token=per_million_tokens(1.50),
            )
        ],
    ),
    # Mistral-Large-3 — same Global Standard rate as the family; explicit key documents the
    # current GA id (matched case-insensitively, so casing here is cosmetic).
    "Mistral-Large-3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.50),
                output_cost_per_token=per_million_tokens(1.50),
            )
        ],
    ),
    # Mistral Medium 3.5 (Preview). Billed via the "MM3.5" Global meter (0.0015/0.0075 per 1K).
    "mistral-medium-3-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.50),
                output_cost_per_token=per_million_tokens(7.50),
            )
        ],
    ),
    "codestral": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(0.90),
            )
        ],
    ),
    # --- OpenAI OSS ---
    "gpt-oss-120b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
            )
        ],
    ),
    # --- Cohere. Lookup is case-insensitive, so the base id resolves regardless of Azure's
    # runtime casing. "Cohere-command-a-plus" is a version-agnostic prefix so future dated
    # snapshots keep the Plus rate instead of falling through to the (pricier) base key. ---
    "Cohere-command-a-plus": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.80),
                output_cost_per_token=per_million_tokens(3.20),
            )
        ],
    ),
    "Cohere-command-a": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
            )
        ],
    ),
    # Cohere Embed v4 text-input rate; image-input is metered separately and not modeled.
    "embed-v-4-0": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.12), output_cost_per_token=0.0)]
    ),
    # --- MoonshotAI Kimi. Runtime `model` ids per the Azure model catalog are Kimi-K2.5 /
    # Kimi-K2.6 / Kimi-K2.7-Code (the "FW-"/"FW Kimi" form is only the Fireworks billing meter,
    # not the deployable id). Azure offers these only as Data Zone deployments; the base rates below
    # are the DZ meter / 1.1, keeping the base-is-global convention used by every other model. Pass
    # deployment_type="data_zone" to bill the exact DZ rate — the default, unqualified call
    # deliberately under-reports ~9% (there is no global tier to fall back to). ---
    "Kimi-K2.5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.60),
                output_cost_per_token=per_million_tokens(3.00),
                cache_read_cost_per_token=per_million_tokens(0.10),
            )
        ],
    ),
    "Kimi-K2.6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.95),
                output_cost_per_token=per_million_tokens(4.00),
                cache_read_cost_per_token=per_million_tokens(0.16),
            )
        ],
    ),
    "Kimi-K2.7-Code": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.95),
                output_cost_per_token=per_million_tokens(4.00),
                cache_read_cost_per_token=per_million_tokens(0.19),
            )
        ],
    ),
    # --- Phi (Microsoft). Keys mirror Azure's `model` casing for readability; lookup is
    # case-insensitive (see resolve_pricing), so the capitalization here is cosmetic. ---
    "Phi-4-mini-reasoning": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
            )
        ],
    ),
    "Phi-4-reasoning-plus": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.125),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "Phi-4-reasoning": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.125),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "Phi-4-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
            )
        ],
    ),
    # Text/image meter. Must precede the broad "Phi-4" key so multimodal ids do not fall back to it;
    # the audio-input meter is priced separately by Azure and is not modeled here.
    "Phi-4-multimodal": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.08),
                output_cost_per_token=per_million_tokens(0.32),
            )
        ],
    ),
    "Phi-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.125),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
}

# Case-insensitive, longest-prefix index. Azure's `model` casing varies by vendor
# (capital Cohere/Phi/Mistral/Kimi ids, lowercase deepseek/grok/gpt), so lookup folds case.
_PRICING_BY_PREFIX = build_pricing_index(_PRICING)

# Known Azure models with no published Global Standard rate (e.g. the grok-4-20 Preview
# variants). Returning None is correct — do NOT let them fall through to a broad prefix
# (e.g. "grok-4") and inherit a fabricated rate. Add real rates once Azure publishes meters.
_UNPRICED_MODELS = frozenset({"grok-4-20-reasoning", "grok-4-20-non-reasoning"})


def calculate_azure_foundry_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an Azure AI Foundry API call. Returns None if model pricing is unknown."""
    if model.lower() in _UNPRICED_MODELS:
        return None
    pricing = resolve_pricing(model, _PRICING_BY_PREFIX)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)


def apply_cost_multiplier(cost: Cost, multiplier: float) -> Cost:
    """Apply a multiplier to all fields in a cost breakdown."""
    return Cost(
        input_cost=cost.input_cost * multiplier,
        output_cost=cost.output_cost * multiplier,
        total_cost=cost.total_cost * multiplier,
        cache_read_cost=cost.cache_read_cost * multiplier if cost.cache_read_cost is not None else None,
        cache_creation_cost=cost.cache_creation_cost * multiplier if cost.cache_creation_cost is not None else None,
    )
