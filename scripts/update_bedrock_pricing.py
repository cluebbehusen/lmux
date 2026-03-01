# TODO: Improve or replace this script; it does not work well

#!/usr/bin/env python3
"""Fetch Bedrock pricing and generate an updated _PRICING dict for cost.py.

Usage:
    python scripts/update_bedrock_pricing.py [--region REGION] [--write]

By default, prints the generated Python code to stdout for review.
With --write, updates packages/lmux-aws-bedrock/src/lmux_aws_bedrock/cost.py in-place.

Requires:
    - AWS CLI configured with credentials
    - curl (for fetching the bulk pricing JSON)

The Pricing API has known data gaps (e.g., missing Anthropic output prices).
The script flags these with TODO comments. Review the output before committing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PRICING_URL_TEMPLATE = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/{region}/index.json"
)

COST_PY = Path(__file__).resolve().parent.parent / "packages/lmux-aws-bedrock/src/lmux_aws_bedrock/cost.py"

# Inference types we care about (standard on-demand, not batch/flex/priority)
STANDARD_INPUT = "input tokens"
STANDARD_OUTPUT = "output tokens"
CACHE_READ = "prompt cache read input tokens"
CACHE_WRITE = "prompt cache write input tokens"


def _normalize_name(name: str) -> str:
    """Normalize a model display name for fuzzy matching.

    Both the Pricing API and list-foundation-models use slightly different
    display names. This strips common suffixes and normalizes casing so
    they can be joined.
    """
    n = name.strip()
    # Lowercase for case-insensitive matching (MiniMax vs Minimax, etc.)
    n = n.lower()
    # Strip parenthetical version suffixes: "Mistral Large (24.02)" → "mistral large"
    n = re.sub(r"\s*\([\d.]+\)\s*$", "", n)
    # Strip trailing "instruct"
    n = re.sub(r"\s+instruct$", "", n)
    # Strip trailing "it" or "pt" (Gemma 3 12B IT → gemma 3 12b)
    n = re.sub(r"\s+(it|pt)$", "", n)
    # Strip "- text" from Titan embedding names
    n = re.sub(r"\s*-\s*text$", "", n)
    # Strip "(dense)" from Qwen names
    n = re.sub(r"\s*\(dense\)\s*$", "", n)
    # Normalize dashes to spaces (Qwen3-Coder → Qwen3 Coder)
    n = n.replace("-", " ")
    # Normalize whitespace
    return re.sub(r"\s+", " ", n)


# Manual aliases for names that can't be normalized algorithmically.
# Maps pricing API name → foundation model catalog name.
_PRICING_NAME_ALIASES: dict[str, str] = {
    "R1": "DeepSeek-R1",
    "TitanEmbeddingsV2-Text-input": "Titan Text Embeddings V2",
    "Titan Embeddings G1 Image": "Titan Multimodal Embeddings G1",
    "Titan Embeddings G1 Text": "Titan Embeddings G1 - Text",
    "Qwen3 Coder 30B A3B": "Qwen3-Coder-30B-A3B-Instruct",
    "Nova 2.0 Lite": "Nova 2 Lite",
    "NVIDIA Nemotron Nano 2": "NVIDIA Nemotron Nano 9B v2",
    "NVIDIA Nemotron Nano 2 VL": "NVIDIA Nemotron Nano 12B v2 VL BF16",
    "Ministral 3B 3.0": "Ministral 3B",
    "Ministral 8B 3.0": "Ministral 3 8B",
    "Voxtral Mini 1.0": "Voxtral Mini 3B 2507",
    "Voxtral Small 1.0": "Voxtral Small 24B 2507",
    "Magistral Small 1.2": "Magistral Small 2509",
    "Devstral": "Devstral 2 123B",
    "Pixtral Large 25.02": "Pixtral Large (25.02)",
}


def fetch_model_catalog(region: str) -> tuple[dict[str, str], dict[str, str]]:
    """Call 'aws bedrock list-foundation-models'.

    Returns:
        (exact_name_to_prefix, normalized_name_to_prefix)
    """
    result = subprocess.run(
        ["aws", "bedrock", "list-foundation-models", "--region", region, "--output", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    # Group by modelName, pick the base model ID (shortest, without context-window suffixes)
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for m in data["modelSummaries"]:
        name_to_ids[m["modelName"]].append(m["modelId"])

    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}

    for name, ids in name_to_ids.items():
        # Filter out context-window variants (e.g., :0:256k, :0:mm)
        base_ids = [mid for mid in ids if mid.count(":") <= 1]
        if not base_ids:
            base_ids = ids
        # Pick the shortest one and strip trailing :0
        shortest = min(base_ids, key=len)
        prefix = re.sub(r":0$", "", shortest)
        # Strip date segments (e.g., -20250514) to get a cleaner prefix
        prefix = re.sub(r"-\d{8}", "", prefix)

        exact[name] = prefix
        normalized[_normalize_name(name)] = prefix

    return exact, normalized


def _resolve_model_id(
    pricing_name: str,
    exact_map: dict[str, str],
    normalized_map: dict[str, str],
) -> str | None:
    """Try to resolve a pricing display name to a model ID prefix."""
    # 1. Check manual aliases
    alias = _PRICING_NAME_ALIASES.get(pricing_name)
    if alias and alias in exact_map:
        return exact_map[alias]

    # 2. Exact match
    if pricing_name in exact_map:
        return exact_map[pricing_name]

    # 3. Normalized match
    norm = _normalize_name(pricing_name)
    if norm in normalized_map:
        return normalized_map[norm]

    return None


def fetch_bulk_pricing(region: str) -> dict[str, dict[str, float]]:
    """Fetch the bulk pricing JSON and return display_name -> {type: price_per_million}."""
    url = PRICING_URL_TEMPLATE.format(region=region)
    result = subprocess.run(
        ["curl", "-sf", url],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})

    # Build SKU -> product attributes lookup
    sku_attrs: dict[str, dict[str, str]] = {}
    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        sku_attrs[sku] = attrs

    # Parse prices
    by_model: dict[str, dict[str, float]] = defaultdict(dict)

    for sku, term_data in terms.items():
        attrs = sku_attrs.get(sku, {})
        if not attrs:
            continue

        model = attrs.get("model") or attrs.get("titanModel", "")
        if not model:
            continue

        inference_type = attrs.get("inferenceType", "").lower().strip()

        # Only standard on-demand types
        if inference_type not in {STANDARD_INPUT, STANDARD_OUTPUT, CACHE_READ, CACHE_WRITE}:
            continue

        for offer in term_data.values():
            for dim in offer.get("priceDimensions", {}).values():
                if dim.get("unit") != "1K tokens":
                    continue
                price_per_1k = float(dim.get("pricePerUnit", {}).get("USD", "0"))
                price_per_million = price_per_1k * 1000
                by_model[model][inference_type] = price_per_million

    return dict(by_model)


def _format_price(price: float) -> str:
    """Format a per-million-token price, dropping unnecessary trailing zeros."""
    if price == 0.0:
        return "0.0"
    # Use enough decimal places to represent the value accurately
    formatted = f"{price:.4f}"
    # Strip trailing zeros but keep at least one decimal place
    formatted = formatted.rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted += ".0"
    return formatted


def generate_pricing_entry(
    model_id: str,
    input_cost: float | None,
    output_cost: float | None,
    cache_read: float | None,
    cache_write: float | None,
) -> str:
    """Generate a single _PRICING dict entry as Python code."""
    lines = [f'    "{model_id}": ModelPricing(']
    lines.append("        tiers=[")
    lines.append("            PricingTier(")

    if input_cost is not None:
        lines.append(f"                input_cost_per_token=per_million_tokens({_format_price(input_cost)}),")
    else:
        lines.append("                input_cost_per_token=0.0,  # TODO: missing from API")

    if output_cost is not None:
        lines.append(f"                output_cost_per_token=per_million_tokens({_format_price(output_cost)}),")
    elif input_cost is not None and input_cost > 0:
        # Embedding models legitimately have 0 output cost
        lines.append("                output_cost_per_token=0.0,  # TODO: missing from API")
    else:
        lines.append("                output_cost_per_token=0.0,")

    if cache_read is not None and cache_read > 0:
        lines.append(f"                cache_read_cost_per_token=per_million_tokens({_format_price(cache_read)}),")
    if cache_write is not None and cache_write > 0:
        lines.append(f"                cache_creation_cost_per_token=per_million_tokens({_format_price(cache_write)}),")

    lines.append("            )")
    lines.append("        ]")
    lines.append("    ),")
    return "\n".join(lines)


def generate_cost_py(entries: list[tuple[str, str]]) -> str:
    """Generate the full cost.py file content."""
    header = '''\
"""AWS Bedrock pricing data and cost calculation.

Prices are for the us-east-1 region (on-demand). Use register_pricing()
on BedrockProvider for overrides or other regions.

Auto-generated by scripts/update_bedrock_pricing.py — manual edits will be overwritten.
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
'''

    footer = '''\
}

# Pre-sorted by key length descending for longest-prefix matching
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_bedrock_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Bedrock API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
'''

    body_parts: list[str] = []
    current_provider: str | None = None
    for provider, entry_code in entries:
        if provider != current_provider:
            if current_provider is not None:
                body_parts.append("")
            body_parts.append(f"    # ── {provider} " + "─" * max(1, 55 - len(provider)))
            current_provider = provider
        body_parts.append(entry_code)

    return header + "\n".join(body_parts) + "\n" + footer


def group_by_provider(model_id: str) -> str:
    """Derive a provider grouping from the model ID prefix."""
    if model_id.startswith("anthropic."):
        return "Anthropic Claude (via Bedrock)"
    if model_id.startswith("amazon.nova"):
        return "Amazon Nova"
    if model_id.startswith("amazon.titan"):
        return "Amazon Titan"
    if model_id.startswith("meta."):
        return "Meta Llama (via Bedrock)"
    if model_id.startswith("mistral."):
        return "Mistral (via Bedrock)"
    if model_id.startswith("cohere."):
        return "Cohere (via Bedrock)"
    if model_id.startswith("deepseek."):
        return "DeepSeek (via Bedrock)"
    return "Other"


def run(region: str, *, write: bool) -> None:
    """Main entry point."""
    print(f"Fetching model catalog from {region}...", file=sys.stderr)
    exact_map, normalized_map = fetch_model_catalog(region)
    print(f"  Found {len(exact_map)} unique model names", file=sys.stderr)

    print(f"Fetching bulk pricing for {region}...", file=sys.stderr)
    pricing = fetch_bulk_pricing(region)
    print(f"  Found pricing for {len(pricing)} models", file=sys.stderr)

    # Join on display name (exact → normalized → alias)
    matched: list[dict[str, Any]] = []
    unmatched_pricing: list[str] = []
    seen_model_ids: set[str] = set()

    for display_name, prices in sorted(pricing.items()):
        model_id = _resolve_model_id(display_name, exact_map, normalized_map)
        if model_id is None:
            unmatched_pricing.append(display_name)
            continue

        # Avoid duplicates (e.g., Titan Text G1 Express/Lite/Premier all aliased to same ID)
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)

        input_cost = prices.get(STANDARD_INPUT)
        output_cost = prices.get(STANDARD_OUTPUT)
        cache_read = prices.get(CACHE_READ)
        cache_write = prices.get(CACHE_WRITE)

        # Skip if we have no useful pricing data
        if input_cost is None and output_cost is None:
            continue

        matched.append(
            {
                "display_name": display_name,
                "model_id": model_id,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cache_read": cache_read,
                "cache_write": cache_write,
            }
        )

    # Report
    print(f"\nMatched {len(matched)} models:", file=sys.stderr)
    for m in matched:
        missing: list[str] = []
        if m["input_cost"] is None:
            missing.append("input")
        if m["output_cost"] is None:
            missing.append("output")
        warn = f"  ⚠ MISSING: {', '.join(missing)}" if missing else ""
        print(f"  {m['model_id']:55s} ← {m['display_name']}{warn}", file=sys.stderr)

    if unmatched_pricing:
        print(f"\nUnmatched pricing entries ({len(unmatched_pricing)}):", file=sys.stderr)
        for name in sorted(unmatched_pricing):
            print(f"  {name}", file=sys.stderr)

    # Generate code
    entries: list[tuple[str, str]] = []
    for m in sorted(matched, key=lambda x: (group_by_provider(x["model_id"]), x["model_id"])):
        provider = group_by_provider(m["model_id"])
        code = generate_pricing_entry(
            m["model_id"],
            m["input_cost"],
            m["output_cost"],
            m["cache_read"],
            m["cache_write"],
        )
        entries.append((provider, code))

    output = generate_cost_py(entries)

    if write:
        COST_PY.write_text(output)
        print(f"\nWrote {COST_PY}", file=sys.stderr)
    else:
        print(output)
        print("\nPass --write to update cost.py in-place", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Bedrock pricing data")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--write", action="store_true", help="Write directly to cost.py")
    args = parser.parse_args()
    run(args.region, write=args.write)
