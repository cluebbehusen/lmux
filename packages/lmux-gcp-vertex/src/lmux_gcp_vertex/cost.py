"""GCP Vertex AI pricing data and cost calculation.

Prices are for standard on-demand (global endpoint) pricing.
Pricing source: https://cloud.google.com/vertex-ai/generative-ai/pricing
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # ── Google Gemini 3 (Preview) ──────────────────────────────
    "gemini-3.1-pro-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.0),
                output_cost_per_token=per_million_tokens(12.0),
                cache_read_cost_per_token=per_million_tokens(0.2),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(4.0),
                output_cost_per_token=per_million_tokens(18.0),
                cache_read_cost_per_token=per_million_tokens(0.4),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "gemini-3-pro-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.0),
                output_cost_per_token=per_million_tokens(12.0),
                cache_read_cost_per_token=per_million_tokens(0.2),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(4.0),
                output_cost_per_token=per_million_tokens(18.0),
                cache_read_cost_per_token=per_million_tokens(0.4),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "gemini-3.1-flash-lite-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.50),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    "gemini-3.1-flash-image-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.50),
                output_cost_per_token=per_million_tokens(3.00),
            ),
        ],
    ),
    "gemini-3-pro-image-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.0),
                output_cost_per_token=per_million_tokens(12.0),
            ),
        ],
    ),
    "gemini-3-flash-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.5),
                output_cost_per_token=per_million_tokens(3.0),
                cache_read_cost_per_token=per_million_tokens(0.05),
            ),
        ],
    ),
    # ── Google Gemini 2.5 ──────────────────────────────────────
    "gemini-2.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.0),
                cache_read_cost_per_token=per_million_tokens(0.125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.5),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.25),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "gemini-2.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.3),
                output_cost_per_token=per_million_tokens(2.5),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.1),
                output_cost_per_token=per_million_tokens(0.4),
                cache_read_cost_per_token=per_million_tokens(0.01),
            ),
        ],
    ),
    # ── Google Gemini 2.0 ──────────────────────────────────────
    "gemini-2.0-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.6),
            ),
        ],
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    # ── Google Gemini 1.5 ──────────────────────────────────────
    "gemini-1.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(5.0),
                cache_read_cost_per_token=per_million_tokens(0.3125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.5),
                output_cost_per_token=per_million_tokens(10.0),
                cache_read_cost_per_token=per_million_tokens(0.625),
                min_input_tokens=128_000,
            ),
        ],
    ),
    "gemini-1.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.3),
                cache_read_cost_per_token=per_million_tokens(0.01875),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.6),
                cache_read_cost_per_token=per_million_tokens(0.0375),
                min_input_tokens=128_000,
            ),
        ],
    ),
    # ── Anthropic Claude (via Vertex AI) ───────────────────────
    "claude-opus-4-7": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.0),
                output_cost_per_token=per_million_tokens(25.0),
                cache_read_cost_per_token=per_million_tokens(0.5),
            ),
        ],
    ),
    "claude-opus-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.0),
                output_cost_per_token=per_million_tokens(25.0),
                cache_read_cost_per_token=per_million_tokens(0.5),
            ),
        ],
    ),
    "claude-sonnet-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-opus-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.0),
                output_cost_per_token=per_million_tokens(25.0),
                cache_read_cost_per_token=per_million_tokens(0.5),
            ),
        ],
    ),
    "claude-sonnet-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.0),
                output_cost_per_token=per_million_tokens(22.5),
                cache_read_cost_per_token=per_million_tokens(0.6),
                min_input_tokens=200000,
            ),
        ],
    ),
    "claude-opus-4-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-opus-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-sonnet-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.0),
                output_cost_per_token=per_million_tokens(22.5),
                cache_read_cost_per_token=per_million_tokens(0.6),
                min_input_tokens=200000,
            ),
        ],
    ),
    "claude-haiku-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.0),
                output_cost_per_token=per_million_tokens(5.0),
                cache_read_cost_per_token=per_million_tokens(0.1),
            ),
        ],
    ),
    "claude-3-5-sonnet-v2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-3-5-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.8),
                output_cost_per_token=per_million_tokens(4.0),
                cache_read_cost_per_token=per_million_tokens(0.08),
            ),
        ],
    ),
    "claude-3-opus": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-3-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.25),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    # ── Mistral (via Vertex AI) ────────────────────────────────
    "mistral-medium-3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.40),
                output_cost_per_token=per_million_tokens(2.00),
            ),
        ],
    ),
    "mistral-small-3.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=per_million_tokens(0.30),
            ),
        ],
    ),
    "codestral-2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(0.90),
            ),
        ],
    ),
    # ── Meta Llama (via Vertex AI) ─────────────────────────────
    "llama-4-maverick-17b-128e-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.35),
                output_cost_per_token=per_million_tokens(1.15),
            ),
        ],
    ),
    "llama-4-scout-17b-16e-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(0.70),
            ),
        ],
    ),
    "llama-3.3-70b-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.72),
                output_cost_per_token=per_million_tokens(0.72),
            ),
        ],
    ),
    # ── DeepSeek (via Vertex AI) ─────────────────────────────────
    "deepseek-v3-2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.56),
                output_cost_per_token=per_million_tokens(1.68),
                cache_read_cost_per_token=per_million_tokens(0.056),
            ),
        ],
    ),
    "deepseek-v3-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.60),
                output_cost_per_token=per_million_tokens(1.70),
                cache_read_cost_per_token=per_million_tokens(0.06),
            ),
        ],
    ),
    "deepseek-r1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.35),
                output_cost_per_token=per_million_tokens(5.40),
            ),
        ],
    ),
    "deepseek-ocr": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(1.20),
            ),
        ],
    ),
    # ── Meta Llama (additional, via Vertex AI) ────────────────
    "llama-3.1-405b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(16.00),
            ),
        ],
    ),
    # ── Mistral (additional, via Vertex AI) ───────────────────
    "mistral-ocr": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.0005),
                output_cost_per_token=per_million_tokens(0.0005),
            ),
        ],
    ),
    # ── Google Gemini (additional) ────────────────────────────
    "gemini-2.5-flash-image": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(2.50),
            ),
        ],
    ),
    "gemini-2.5-pro-computer-use-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
            ),
        ],
    ),
    # ── xAI Grok (via Vertex AI) ───────────────────────────────
    "grok-4-20-non-reasoning": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(6.00),
                cache_read_cost_per_token=per_million_tokens(0.20),
            ),
        ],
    ),
    "grok-4-1-fast-reasoning": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
                cache_read_cost_per_token=per_million_tokens(0.05),
            ),
        ],
    ),
    "grok-4-1-fast-non-reasoning": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
                cache_read_cost_per_token=per_million_tokens(0.05),
            ),
        ],
    ),
    # ── MiniMax (via Vertex AI) ────────────────────────────────
    "minimax-m2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(1.20),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    # ── Moonshot AI (via Vertex AI) ────────────────────────────
    "kimi-k2-thinking": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.60),
                output_cost_per_token=per_million_tokens(2.50),
                cache_read_cost_per_token=per_million_tokens(0.06),
            ),
        ],
    ),
    # ── Qwen (via Vertex AI) ───────────────────────────────────
    "qwen3-coder-480b-a35b-instruct": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.22),
                output_cost_per_token=per_million_tokens(1.80),
                cache_read_cost_per_token=per_million_tokens(0.022),
            ),
        ],
    ),
    "qwen3-next-80b-thinking": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(1.20),
            ),
        ],
    ),
    "qwen3-next-80b-instruct": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(1.20),
            ),
        ],
    ),
    "qwen3-235b-a22b-instruct-2507": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.22),
                output_cost_per_token=per_million_tokens(0.88),
            ),
        ],
    ),
    # ── Zhipu AI GLM (via Vertex AI) ───────────────────────────
    "glm-4.7": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.60),
                output_cost_per_token=per_million_tokens(2.20),
            ),
        ],
    ),
    "glm-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(3.20),
                cache_read_cost_per_token=per_million_tokens(0.10),
            ),
        ],
    ),
    # ── OpenAI gpt-oss (via Vertex AI) ─────────────────────────
    "gpt-oss-120b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.09),
                output_cost_per_token=per_million_tokens(0.36),
            ),
        ],
    ),
    "gpt-oss-20b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.07),
                output_cost_per_token=per_million_tokens(0.25),
                cache_read_cost_per_token=per_million_tokens(0.007),
            ),
        ],
    ),
    # ── Google Gemma (via Vertex AI) ───────────────────────────
    "gemma-4-26b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
            ),
        ],
    ),
    # ── Embedding models ───────────────────────────────────────
    "gemini-embedding-2-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "gemini-embedding-001": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-embedding-005": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-embedding-004": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-multilingual-embedding-002": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
}

# Pre-sorted by key length descending for prefix matching (longest match first)
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_gcp_vertex_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Vertex AI API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
