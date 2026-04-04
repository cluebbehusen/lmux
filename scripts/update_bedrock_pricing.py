#!/usr/bin/env python3
"""Generate AWS Bedrock cost.py from the public AWS Pricing API.

Fetches pricing from two unauthenticated API endpoints:
- AmazonBedrock: third-party models (DeepSeek, Gemma, Mistral, etc.)
- AmazonBedrockFoundationModels: Claude, Amazon Nova/Titan, Cohere, etc.

Usage:
    python3 scripts/update_bedrock_pricing.py               # stdout, us-east-1 only
    python3 scripts/update_bedrock_pricing.py --write        # overwrite cost.py
    python3 scripts/update_bedrock_pricing.py --regions eu-west-1 ap-northeast-1
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

API_BASE = "https://pricing.us-east-1.amazonaws.com"
DEFAULT_REGION = "us-east-1"
COST_PY_PATH = (
    Path(__file__).resolve().parent.parent / "packages" / "lmux-aws-bedrock" / "src" / "lmux_aws_bedrock" / "cost.py"
)

# Long-context tier threshold (tokens). Anthropic uses 200K for all models.
LCTX_THRESHOLD = 200_000

# ── Model ID mappings ────────────────────────────────────────────────────────

# Foundation Models API: servicename (after stripping " (Amazon Bedrock Edition)") -> Bedrock model ID
FM_SERVICENAME_MAP: dict[str, str] = {
    # Anthropic Claude
    "Claude Opus 4.6": "anthropic.claude-opus-4-6-v1",
    "Claude Sonnet 4.6": "anthropic.claude-sonnet-4-6-v1",
    "Claude Opus 4.5": "anthropic.claude-opus-4-5-v1",
    "Claude Sonnet 4.5": "anthropic.claude-sonnet-4-5-v1",
    "Claude Haiku 4.5": "anthropic.claude-haiku-4-5-v1",
    "Claude Sonnet 4": "anthropic.claude-sonnet-4-v1",
    "Claude Opus 4": "anthropic.claude-opus-4-v1",
    "Claude Opus 4.1": "anthropic.claude-opus-4-1-v1",
    "Claude 3.7 Sonnet": "anthropic.claude-3-7-sonnet-v1",
    "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-v2",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-v1",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-v1",
    "Claude 3 Opus": "anthropic.claude-3-opus-v1",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-v1",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-v1",
    "Claude": "anthropic.claude-v2",
    "Claude Instant": "anthropic.claude-instant-v1",
    # Cohere
    "Cohere Command R": "cohere.command-r-v1",
    "Cohere Command R+": "cohere.command-r-plus-v1",
    "Cohere Embed 3 Model - English": "cohere.embed-english-v3",
    "Cohere Embed Model 3 - Multilingual": "cohere.embed-multilingual-v3",
    "Cohere Embed 4 Model": "cohere.embed-v4",
    "Cohere Generate Model - Command": "cohere.command-text-v14",
    "Cohere Generate Model - Command-Light": "cohere.command-light-text-v14",
    # AI21 Labs
    "Jamba 1.5 Large": "ai21.jamba-1-5-large-v1",
    "Jamba 1.5 Mini": "ai21.jamba-1-5-mini-v1",
    "Jamba-Instruct": "ai21.jamba-instruct-v1",
    "Jurassic-2 Mid": "ai21.j2-mid-v1",
    "Jurassic-2 Ultra": "ai21.j2-ultra-v1",
    # Meta Llama 2 (via Foundation Models)
    "Meta Llama 2 Chat 13B": "meta.llama2-13b-chat-v1",
    "Meta Llama 2 Chat 70B": "meta.llama2-70b-chat-v1",
    # Writer
    "Palmyra X4": "writer.palmyra-x4-v1",
    "Palmyra X5": "writer.palmyra-x5-v1",
}

# AmazonBedrock API non-mantle: model attribute value -> Bedrock model ID
NON_MANTLE_MODEL_MAP: dict[str, str] = {
    # Amazon Nova
    "Nova Micro": "amazon.nova-micro-v1",
    "Nova Lite": "amazon.nova-lite-v1",
    "Nova Pro": "amazon.nova-pro-v1",
    "Nova Premier": "amazon.nova-premier-v1",
    "Nova 2.0 Lite": "amazon.nova-2-lite-v1",
    "Nova 2.0 Pro": "amazon.nova-2-pro-v1",
    "Nova 2.0 Omni": "amazon.nova-2-omni-v1",
    # DeepSeek (R1 is non-mantle only; v3.x is mantle)
    "R1": "deepseek.r1-v1",
    "DeepSeek v3.2": "deepseek.v3.2",
    # Meta Llama (3.x+ are non-mantle only)
    "Llama 3 8B": "meta.llama3-8b-instruct-v1",
    "Llama 3 70B": "meta.llama3-70b-instruct-v1",
    "Llama 3.1 8B": "meta.llama3-1-8b-instruct-v1",
    "Llama 3.1 70B": "meta.llama3-1-70b-instruct-v1",
    "Llama 3.2 1B": "meta.llama3-2-1b-instruct-v1",
    "Llama 3.2 3B": "meta.llama3-2-3b-instruct-v1",
    "Llama 3.2 11B": "meta.llama3-2-11b-instruct-v1",
    "Llama 3.2 90B": "meta.llama3-2-90b-instruct-v1",
    "Llama 3.3 70B": "meta.llama3-3-70b-instruct-v1",
    "Llama 4 Maverick 17B": "meta.llama4-maverick-17b-instruct-v1",
    "Llama 4 Scout 17B": "meta.llama4-scout-17b-instruct-v1",
    # Old Mistral (non-mantle only; newer ones are mantle)
    "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",
    "Mixtral 8x7B": "mistral.mixtral-8x7b-instruct-v0:1",
    "Mistral Large": "mistral.mistral-large-2402-v1",
    "Mistral Small": "mistral.mistral-small-2402-v1",
    "Mistral Large 3": "mistral.mistral-large-3-675b-instruct",
}

# For entries with empty model attribute: usagetype key -> Bedrock model ID
USAGETYPE_KEY_MAP: dict[str, str] = {
    "TitanEmbeddingV2-Text": "amazon.titan-embed-text-v2",
    "TitanEmbeddingsG1-Text": "amazon.titan-embed-text-v1",
    "TitanEmbeddingsG1-Image": "amazon.titan-embed-image-v1",
    "TitanTextG1-Express": "amazon.titan-text-express-v1",
    "TitanTextG1-Lite": "amazon.titan-text-lite-v1",
    "TitanText-Premier": "amazon.titan-text-premier-v1",
}

# Provider groups for comment headers in generated code
PROVIDER_GROUPS: list[tuple[str, str]] = [
    ("amazon.", "Amazon Nova / Titan"),
    ("ai21.", "AI21 Labs"),
    ("anthropic.", "Anthropic Claude (via Bedrock)"),
    ("cohere.", "Cohere (via Bedrock)"),
    ("deepseek.", "DeepSeek (via Bedrock)"),
    ("google.", "Google (via Bedrock)"),
    ("meta.", "Meta Llama (via Bedrock)"),
    ("minimax.", "MiniMax (via Bedrock)"),
    ("mistral.", "Mistral (via Bedrock)"),
    ("moonshotai.", "Moonshot (via Bedrock)"),
    ("nvidia.", "Nvidia (via Bedrock)"),
    ("openai.", "OpenAI (via Bedrock)"),
    ("qwen.", "Qwen (via Bedrock)"),
    ("writer.", "Writer (via Bedrock)"),
    ("zai.", "Zhipu AI (via Bedrock)"),
]

# Embedding models (output_cost_per_token = 0.0)
EMBEDDING_PREFIXES = ("amazon.titan-embed", "amazon.nova-embed", "cohere.embed")

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ModelPrices:
    """Intermediate representation of a model's pricing (per million tokens)."""

    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    cache_read_cost: Decimal | None = None
    cache_write_cost: Decimal | None = None
    # Long-context tier (>200K tokens)
    lctx_input_cost: Decimal | None = None
    lctx_output_cost: Decimal | None = None
    lctx_cache_read_cost: Decimal | None = None
    lctx_cache_write_cost: Decimal | None = None

    @property
    def has_lctx(self) -> bool:
        return self.lctx_input_cost is not None


# ── Fetching ─────────────────────────────────────────────────────────────────


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON from a URL."""
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read())


def fetch_region_index(service: str) -> dict[str, str]:
    """Return {region_code: version_url} for a pricing service."""
    data = _fetch_json(f"{API_BASE}/offers/v1.0/aws/{service}/current/region_index.json")
    return {code: info["currentVersionUrl"] for code, info in data.get("regions", {}).items()}


def fetch_pricing(service: str, region: str) -> dict[str, Any]:
    """Fetch full pricing data for a service+region."""
    index = fetch_region_index(service)
    if region not in index:
        _warn(f"Region {region} not found for {service}")
        return {}
    url = API_BASE + index[region]
    return _fetch_json(url)


# ── Parsing: AmazonBedrock mantle models ─────────────────────────────────────


def parse_mantle_models(data: dict[str, Any]) -> dict[str, ModelPrices]:
    """Parse mantle entries from AmazonBedrock API. Prices are per 1K tokens."""
    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})
    result: dict[str, ModelPrices] = {}

    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        ut = attrs.get("usagetype", "")
        if "-mantle-" not in ut or not ut.endswith("-standard"):
            continue

        # Extract model ID and dimension from usagetype
        # Pattern: {REGION_PREFIX}-{model_id}-mantle-{dimension}-standard
        mantle_idx = ut.index("-mantle-")
        prefix_end = ut.index("-") + 1  # skip region prefix like "USE1-"
        model_id = ut[prefix_end:mantle_idx]
        dimension_part = ut[mantle_idx + len("-mantle-") : -len("-standard")]

        price = _get_price(sku, terms)
        if price is None:
            continue

        # Prices are per 1K tokens -> multiply by 1000 to get per-M
        price_per_m = price * 1000

        if model_id not in result:
            result[model_id] = ModelPrices()

        mp = result[model_id]
        if dimension_part == "input-tokens":
            mp.input_cost = price_per_m
        elif dimension_part == "output-tokens":
            mp.output_cost = price_per_m
        elif dimension_part == "cache-read-input-tokens":
            mp.cache_read_cost = price_per_m
        elif dimension_part == "cache-write-input-tokens":
            mp.cache_write_cost = price_per_m

    # Remove models with incomplete pricing
    return {k: v for k, v in result.items() if v.input_cost is not None}


# ── Parsing: AmazonBedrock non-mantle models (Amazon Nova/Titan, legacy) ─────


def _classify_dimension(inf_type: str, usagetype: str) -> str | None:
    """Map inferenceType (or usagetype fallback) to a dimension key."""
    # Try inferenceType first, fall back to usagetype
    text = inf_type.lower() if inf_type else usagetype.lower()
    if "cache" in text and "read" in text:
        return "cache_read"
    if "cache" in text and "write" in text:
        return "cache_write"
    if "input" in text:
        return "input"
    if "output" in text:
        return "output"
    return None


def _should_skip_usagetype(ut: str) -> bool:
    """Whether to skip a non-mantle usagetype."""
    ut_lower = ut.lower()
    skip_keywords = [
        "batch",
        "flex",
        "priority",
        "latency-optimized",
        "custom-model",
        "video",
        "audio",
        "speech",
        "training",
        "customization",
        "storage",
        # Image generation (but NOT image embeddings which use "input-tokens")
        "input-image",
        "output-image",
        "created_image",
        "created-image",
        "t2i-",
        "i2i-",
    ]
    return any(kw in ut_lower for kw in skip_keywords)


def _is_global_usagetype(ut: str) -> bool:
    """Whether this usagetype is cross-region global pricing."""
    return "cross-region-global" in ut.lower()


def _resolve_non_mantle_model_id(model_name: str, usagetype: str) -> str | None:
    """Resolve a non-mantle entry to a Bedrock model ID."""
    if model_name and model_name in NON_MANTLE_MODEL_MAP:
        return NON_MANTLE_MODEL_MAP[model_name]
    if not model_name:
        for key, mid in USAGETYPE_KEY_MAP.items():
            if key in usagetype:
                return mid
    return None


def _set_dimension(mp: ModelPrices, dim_name: str, price: Decimal) -> None:
    """Set a price on a ModelPrices by dimension name."""
    if dim_name == "input":
        mp.input_cost = price
    elif dim_name == "output":
        mp.output_cost = price
    elif dim_name == "cache_read":
        mp.cache_read_cost = price
    elif dim_name == "cache_write":
        mp.cache_write_cost = price


def parse_amazon_models(data: dict[str, Any]) -> dict[str, ModelPrices]:
    """Parse non-mantle entries from AmazonBedrock API for Amazon + legacy models."""
    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})

    # Collect prices keyed by model_id -> dimension -> {is_global: price}
    collected: dict[str, dict[str, dict[bool, Decimal]]] = {}

    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        ut = attrs.get("usagetype", "")

        if "-mantle-" in ut or _should_skip_usagetype(ut):
            continue

        dimension = _classify_dimension(attrs.get("inferenceType", ""), ut)
        if dimension is None:
            continue

        model_id = _resolve_non_mantle_model_id(attrs.get("model", "").strip(), ut)
        if model_id is None:
            continue

        price = _get_price(sku, terms)
        if price is None:
            continue

        price_per_m = price * 1000  # per 1K tokens -> per-M
        collected.setdefault(model_id, {}).setdefault(dimension, {})[_is_global_usagetype(ut)] = price_per_m

    # Build result: prefer global prices when available
    result: dict[str, ModelPrices] = {}
    for model_id, dims in collected.items():
        mp = ModelPrices()
        for dim_name, prices_by_scope in dims.items():
            price = prices_by_scope.get(True) or prices_by_scope.get(False)
            if price is not None:
                _set_dimension(mp, dim_name, price)
        if mp.input_cost is not None:
            result[model_id] = mp

    return result


# ── Parsing: AmazonBedrockFoundationModels ───────────────────────────────────


def _parse_fm_dimension(usagetype: str) -> tuple[str, bool] | None:
    """Extract (dimension, is_lctx) from a Foundation Models usagetype.

    Usagetype format: {REGION}-MP:{REGION}_{Dimension}[-{Variant}]-Units

    Returns None for usagetypes we don't care about.
    """
    # Split on the MP: prefix to get the dimension part
    if "-MP:" not in usagetype:
        return None

    dim_part = usagetype.split("-MP:")[1]
    # Strip the region prefix: "USE1_InputTokenCount_Global-Units" -> "InputTokenCount_Global"
    parts = dim_part.split("_", 1)
    if len(parts) < 2:  # noqa: PLR2004
        return None
    field = parts[1].removesuffix("-Units")

    # Skip types we don't handle
    skip_patterns = [
        "ProvisionedThroughput",
        "Reserved",
        "Batch",
        "LatencyOptimized",
        "ModelStorage",
        "Customization",
        "search_units",
        "MillionBatch",
        "CacheWrite1h",
        "Created_image",
        "created_image",
        "inputAudioSecond",
        "inputVideoSecond",
        "InputImageCount",
        "inputTextRequestCount",
        "Cohere_Embed",
    ]
    if any(p in field for p in skip_patterns):
        return None

    is_lctx = "_LCtx" in field

    # Determine dimension (order matters: CacheRead before Input)
    dimension_patterns = [
        ("CacheReadInputTokenCount", "cache_read"),
        ("CacheWriteInputTokenCount", "cache_write"),
        ("InputTokenCount", "input"),
        ("OutputTokenCount", "output"),
    ]
    return next(
        ((dim, is_lctx) for pattern, dim in dimension_patterns if pattern in field),
        None,
    )


def _is_global_fm(usagetype: str) -> bool:
    """Whether this FM usagetype is Global (cross-region) pricing."""
    dim_part = usagetype.split("-MP:")[1] if "-MP:" in usagetype else ""
    # Check for _Global but NOT _Global_Batch
    return "_Global" in dim_part and "_Batch" not in dim_part


def parse_foundation_models(data: dict[str, Any]) -> dict[str, ModelPrices]:
    """Parse AmazonBedrockFoundationModels API. Prices are per million tokens."""
    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})

    # Collect: model_id -> {(dimension, is_lctx, is_global): price}
    collected: dict[str, dict[tuple[str, bool, bool], Decimal]] = {}
    unmapped: set[str] = set()

    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        ut = attrs.get("usagetype", "")
        if "ProvisionedThroughput" in ut:
            continue

        parsed = _parse_fm_dimension(ut)
        if parsed is None:
            continue
        dimension, is_lctx = parsed

        clean_name = attrs.get("servicename", "").replace(" (Amazon Bedrock Edition)", "")
        if "(100K)" in clean_name:
            continue

        model_id = FM_SERVICENAME_MAP.get(clean_name)
        if model_id is None:
            unmapped.add(clean_name)
            continue

        price = _get_price(sku, terms)
        if price is None:
            continue

        # Prices are already per million tokens
        collected.setdefault(model_id, {})[(dimension, is_lctx, _is_global_fm(ut))] = price

    if unmapped:
        _warn(f"Unmapped Foundation Models servicenames: {sorted(unmapped)}")

    return _build_fm_result(collected)


def _build_fm_result(
    collected: dict[str, dict[tuple[str, bool, bool], Decimal]],
) -> dict[str, ModelPrices]:
    """Build ModelPrices from collected Foundation Models data, preferring Global prices."""
    result: dict[str, ModelPrices] = {}
    for model_id, prices in collected.items():
        mp = ModelPrices()
        for dim_name in ("input", "output", "cache_read", "cache_write"):
            std = prices.get((dim_name, False, True)) or prices.get((dim_name, False, False))
            lctx = prices.get((dim_name, True, True)) or prices.get((dim_name, True, False))
            if std is not None:
                _set_dimension(mp, dim_name, std)
            _set_fm_lctx(mp, dim_name, lctx)
        if mp.input_cost is not None:
            result[model_id] = mp
    return result


def _set_fm_lctx(mp: ModelPrices, dim_name: str, price: Decimal | None) -> None:
    """Set a long-context tier price on a ModelPrices."""
    if dim_name == "input":
        mp.lctx_input_cost = price
    elif dim_name == "output":
        mp.lctx_output_cost = price
    elif dim_name == "cache_read":
        mp.lctx_cache_read_cost = price
    elif dim_name == "cache_write":
        mp.lctx_cache_write_cost = price


# ── Merging ──────────────────────────────────────────────────────────────────


def merge_pricing(
    mantle: dict[str, ModelPrices],
    amazon: dict[str, ModelPrices],
    foundation: dict[str, ModelPrices],
) -> dict[str, ModelPrices]:
    """Merge pricing from all three sources.

    Priority: Foundation Models > mantle > amazon (non-mantle).
    Foundation Models has cache pricing and LCtx tiers for Claude.
    Mantle has exact model IDs for third-party models.
    Amazon (non-mantle) fills in Nova/Titan and legacy models.
    """
    result: dict[str, ModelPrices] = {}

    # Start with amazon (non-mantle) as the base
    result.update(amazon)

    # Layer mantle on top (higher priority for third-party models)
    result.update(mantle)

    # Layer Foundation Models on top (highest priority for Claude, etc.)
    result.update(foundation)

    return result


# ── Regional pricing ─────────────────────────────────────────────────────────


def compute_regional_diffs(default: dict[str, ModelPrices], regional: dict[str, ModelPrices]) -> dict[str, ModelPrices]:
    """Return only models whose regional pricing differs from the default."""
    diffs: dict[str, ModelPrices] = {}
    for model_id, reg_prices in regional.items():
        def_prices = default.get(model_id)
        if def_prices is None or _prices_differ(def_prices, reg_prices):
            diffs[model_id] = reg_prices
    return diffs


def _prices_differ(a: ModelPrices, b: ModelPrices) -> bool:
    """Whether two ModelPrices differ in any field (standard or long-context tier)."""
    return (
        a.input_cost != b.input_cost
        or a.output_cost != b.output_cost
        or a.cache_read_cost != b.cache_read_cost
        or a.cache_write_cost != b.cache_write_cost
        or a.lctx_input_cost != b.lctx_input_cost
        or a.lctx_output_cost != b.lctx_output_cost
        or a.lctx_cache_read_cost != b.lctx_cache_read_cost
        or a.lctx_cache_write_cost != b.lctx_cache_write_cost
    )


# ── Code generation ──────────────────────────────────────────────────────────


def _get_provider_group(model_id: str) -> str:
    """Get the provider group name for a model ID."""
    for prefix, group_name in PROVIDER_GROUPS:
        if model_id.startswith(prefix):
            return group_name
    return "Other"


def _fmt(price: Decimal) -> str:
    """Format a Decimal price for code output. Strips trailing zeros but keeps at least one decimal."""
    # Quantize to 6 decimal places max, then strip trailing zeros
    price = price.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP).normalize()
    # Convert to string, avoiding scientific notation
    s = format(price, "f")
    if "." not in s:
        s += ".0"
    # Strip trailing zeros but keep at least one decimal
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return s


def _is_embedding(model_id: str) -> bool:
    """Whether this model is an embedding model (no output cost)."""
    return any(model_id.startswith(p) for p in EMBEDDING_PREFIXES)


def generate_cost_py(
    pricing: dict[str, ModelPrices],
    regional: dict[str, dict[str, ModelPrices]] | None = None,
) -> str:
    """Generate the complete cost.py source code."""
    lines: list[str] = []
    _emit_header(lines, has_regional=bool(regional))
    _emit_pricing_dict(lines, pricing)
    _emit_regional_dict(lines, regional)
    _emit_function(lines)
    return "\n".join(lines)


def _emit_header(lines: list[str], *, has_regional: bool) -> None:
    """Emit module docstring and imports."""
    lines.append('"""AWS Bedrock pricing data and cost calculation.')
    lines.append("")
    lines.append("Prices are for the us-east-1 region (on-demand, cross-region global inference).")
    if has_regional:
        lines.append("Regional pricing overrides are included for regions where prices differ.")
    else:
        lines.append("Use register_pricing() on BedrockProvider for overrides or other regions.")
    lines.append("")
    lines.append("Auto-generated by scripts/update_bedrock_pricing.py -- do not edit manually.")
    lines.append("")
    lines.append("Pricing source: https://aws.amazon.com/bedrock/pricing/")
    lines.append('"""')
    lines.append("")
    lines.append("from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens")
    lines.append("from lmux.types import Cost, Usage")
    lines.append("")


def _emit_pricing_dict(lines: list[str], pricing: dict[str, ModelPrices]) -> None:
    """Emit the _PRICING dict grouped by provider."""
    lines.append("_PRICING: dict[str, ModelPricing] = {")

    groups: dict[str, list[str]] = {}
    for model_id in sorted(pricing.keys()):
        groups.setdefault(_get_provider_group(model_id), []).append(model_id)

    for group_name in [g for _, g in PROVIDER_GROUPS if g in groups] + (["Other"] if "Other" in groups else []):
        if group_name not in groups:
            continue
        lines.append(f"    # -- {group_name} " + "-" * (56 - len(group_name)))
        for model_id in groups[group_name]:
            _emit_model_pricing(lines, model_id, pricing[model_id])

    lines.append("}")
    lines.append("")


def _emit_regional_dict(
    lines: list[str],
    regional: dict[str, dict[str, ModelPrices]] | None,
) -> None:
    """Emit the _REGIONAL_PRICING dict and prefix list."""
    if regional:
        lines.append("# Regional pricing overrides (only models that differ from us-east-1)")
        lines.append("_REGIONAL_PRICING: dict[str, dict[str, ModelPricing]] = {")
        for region in sorted(regional.keys()):
            if not regional[region]:
                continue
            lines.append(f'    "{region}": {{')
            for model_id in sorted(regional[region].keys()):
                _emit_model_pricing(lines, model_id, regional[region][model_id], indent=8)
            lines.append("    },")
        lines.append("}")
        lines.append("")
    else:
        lines.append("_REGIONAL_PRICING: dict[str, dict[str, ModelPricing]] = {}")
        lines.append("")

    lines.append("# Pre-sorted by key length descending for longest-prefix matching")
    lines.append("_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)")
    lines.append("")
    lines.append("")


_FUNCTION_BODY = """\
def calculate_bedrock_cost(model: str, usage: Usage, *, region: str | None = None) -> Cost | None:
    \"\"\"Calculate cost for a Bedrock API call. Returns None if model pricing is unknown.\"\"\"
    # Try regional pricing first if region specified
    if region is not None and region != "us-east-1":
        regional = _REGIONAL_PRICING.get(region, {})
        pricing = regional.get(model)
        if pricing is None:
            for prefix, p in sorted(regional.items(), key=lambda item: len(item[0]), reverse=True):
                if model.startswith(prefix):
                    pricing = p
                    break
        if pricing is not None:
            return calculate_cost(usage, pricing)

    # Fall back to default (us-east-1) pricing
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
"""


def _emit_function(lines: list[str]) -> None:
    """Emit the calculate_bedrock_cost function."""
    lines.extend(_FUNCTION_BODY.splitlines())
    lines.append("")


def _emit_model_pricing(lines: list[str], model_id: str, mp: ModelPrices, indent: int = 4) -> None:
    """Emit a single ModelPricing entry."""
    pad = " " * indent
    is_emb = _is_embedding(model_id)

    lines.append(f'{pad}"{model_id}": ModelPricing(')
    lines.append(f"{pad}    tiers=[")

    # Standard tier
    _emit_tier(lines, mp, is_emb, indent + 8, is_lctx=False)

    # Long-context tier (if present)
    if mp.has_lctx:
        _emit_tier(lines, mp, is_emb, indent + 8, is_lctx=True)

    lines.append(f"{pad}    ],")
    lines.append(f"{pad}),")


def _emit_tier(lines: list[str], mp: ModelPrices, is_emb: bool, indent: int, *, is_lctx: bool) -> None:
    """Emit a single PricingTier."""
    pad = " " * indent
    lines.append(f"{pad}PricingTier(")

    if is_lctx:
        input_cost = mp.lctx_input_cost
        output_cost = mp.lctx_output_cost
        cache_read = mp.lctx_cache_read_cost
        cache_write = mp.lctx_cache_write_cost
    else:
        input_cost = mp.input_cost
        output_cost = mp.output_cost
        cache_read = mp.cache_read_cost
        cache_write = mp.cache_write_cost

    if input_cost is not None:
        lines.append(f"{pad}    input_cost_per_token=per_million_tokens({_fmt(input_cost)}),")
    if is_emb:
        lines.append(f"{pad}    output_cost_per_token=0.0,")
    elif output_cost is not None:
        lines.append(f"{pad}    output_cost_per_token=per_million_tokens({_fmt(output_cost)}),")
    if cache_read is not None and cache_read > 0:
        lines.append(f"{pad}    cache_read_cost_per_token=per_million_tokens({_fmt(cache_read)}),")
    if cache_write is not None and cache_write > 0:
        lines.append(f"{pad}    cache_creation_cost_per_token=per_million_tokens({_fmt(cache_write)}),")
    if is_lctx:
        lines.append(f"{pad}    min_input_tokens={LCTX_THRESHOLD},")

    lines.append(f"{pad}),")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_price(sku: str, terms: dict[str, Any]) -> Decimal | None:
    """Extract the USD price from OnDemand terms for a SKU."""
    if sku not in terms:
        return None
    for offer in terms[sku].values():
        for dim in offer.get("priceDimensions", {}).values():
            usd = dim.get("pricePerUnit", {}).get("USD")
            if usd is not None:
                return Decimal(usd)
    return None


def _warn(msg: str) -> None:
    """Print a warning to stderr."""
    print(f"WARNING: {msg}", file=sys.stderr)  # noqa: T201


def _fetch_regional_diffs(
    args: argparse.Namespace,
    default_pricing: dict[str, ModelPrices],
) -> dict[str, dict[str, ModelPrices]] | None:
    """Fetch pricing for requested regions and return diffs from us-east-1."""
    if not args.regions and not args.all_regions:
        return None

    if args.all_regions:
        bedrock_index = fetch_region_index("AmazonBedrock")
        fm_index = fetch_region_index("AmazonBedrockFoundationModels")
        regions = sorted(set(bedrock_index.keys()) | set(fm_index.keys()))
        regions = [r for r in regions if r != DEFAULT_REGION]
    else:
        regions = [r for r in args.regions if r != DEFAULT_REGION]

    regional_diffs: dict[str, dict[str, ModelPrices]] = {}
    for region in regions:
        _info(f"Fetching pricing for {region}...")
        try:
            reg_bedrock = fetch_pricing("AmazonBedrock", region)
            reg_fm = fetch_pricing("AmazonBedrockFoundationModels", region)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            _warn(f"Failed to fetch {region}: {e}")
            continue

        reg_mantle = parse_mantle_models(reg_bedrock)
        reg_amazon = parse_amazon_models(reg_bedrock)
        reg_foundation = parse_foundation_models(reg_fm)
        reg_merged = merge_pricing(reg_mantle, reg_amazon, reg_foundation)

        diffs = compute_regional_diffs(default_pricing, reg_merged)
        if diffs:
            regional_diffs[region] = diffs
            _info(f"  {region}: {len(diffs)} models differ from us-east-1")
        else:
            _info(f"  {region}: all prices match us-east-1")

    return regional_diffs or None


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bedrock cost.py from the AWS Pricing API.",
    )
    _ = parser.add_argument(
        "--write",
        action="store_true",
        help="Write directly to cost.py (default: print to stdout)",
    )
    _ = parser.add_argument(
        "--regions",
        nargs="+",
        metavar="REGION",
        help="Include regional pricing overrides for these regions",
    )
    _ = parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Include all available regions",
    )
    args = parser.parse_args()

    # Fetch us-east-1 data
    _info("Fetching AmazonBedrock pricing for us-east-1...")
    bedrock_data = fetch_pricing("AmazonBedrock", DEFAULT_REGION)
    _info("Fetching AmazonBedrockFoundationModels pricing for us-east-1...")
    fm_data = fetch_pricing("AmazonBedrockFoundationModels", DEFAULT_REGION)

    # Parse
    _info("Parsing mantle models...")
    mantle = parse_mantle_models(bedrock_data)
    _info(f"  Found {len(mantle)} mantle models")

    _info("Parsing Amazon models (Nova/Titan/legacy)...")
    amazon = parse_amazon_models(bedrock_data)
    _info(f"  Found {len(amazon)} Amazon/legacy models")

    _info("Parsing Foundation Models (Claude/Cohere/etc)...")
    foundation = parse_foundation_models(fm_data)
    _info(f"  Found {len(foundation)} Foundation Models")

    # Merge
    default_pricing = merge_pricing(mantle, amazon, foundation)
    _info(f"Total models after merge: {len(default_pricing)}")

    # Regional pricing
    regional_diffs = _fetch_regional_diffs(args, default_pricing)

    # Generate code
    code = generate_cost_py(default_pricing, regional_diffs)

    if args.write:
        _ = COST_PY_PATH.write_text(code)
        _info(f"Wrote {COST_PY_PATH}")
    else:
        print(code)  # noqa: T201


def _info(msg: str) -> None:
    """Print an info message to stderr."""
    print(msg, file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
