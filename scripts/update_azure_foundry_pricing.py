#!/usr/bin/env python3
"""Fetch Azure AI Foundry (Azure OpenAI) pricing and generate an updated _PRICING dict for cost.py.

Usage:
    python scripts/update_azure_foundry_pricing.py [--write]

By default, prints the generated Python code to stdout for review.
With --write, updates packages/lmux-azure-foundry/src/lmux_azure_foundry/cost.py in-place.

Requires:
    - Azure CLI (``az``) configured with credentials
    - ``az`` must be able to access the Azure Retail Prices API

Unlike the Bedrock pricing script, this uses the public Azure Retail Prices
REST API (https://prices.azure.com/api/retail/prices) which requires no
authentication and returns clean, structured pricing data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

COST_PY = Path(__file__).resolve().parent.parent / "packages/lmux-azure-foundry/src/lmux_azure_foundry/cost.py"

# Azure Retail Prices API — public, no auth required
PRICES_API = "https://prices.azure.com/api/retail/prices"

# OData filter for Azure OpenAI Service pricing (Global Standard)
ODATA_FILTER = "serviceName eq 'Azure OpenAI' and priceType eq 'Consumption' and contains(skuName, 'Global-Standard')"


def fetch_prices() -> list[dict[str, Any]]:
    """Fetch all Azure OpenAI pricing entries from the Retail Prices API."""
    all_items: list[dict[str, Any]] = []
    url = f"{PRICES_API}?$filter={ODATA_FILTER}"

    while url:
        result = subprocess.run(
            ["curl", "-sf", url],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        all_items.extend(data.get("Items", []))
        url = data.get("NextPageLink")
        if url:
            print(f"  Fetched {len(all_items)} items so far...", file=sys.stderr)

    return all_items


def _classify_meter(meter_name: str) -> str | None:
    """Classify a meter name into input/output/cache_read/cache_write or None."""
    lower = meter_name.lower()
    if "cached input" in lower or "cache read" in lower:
        return "cache_read"
    if "input" in lower:
        return "input"
    if "output" in lower:
        return "output"
    return None


def _extract_model_name(meter_name: str) -> str | None:
    """Extract the model name from a meter name like 'GPT-4o Input Tokens'."""
    lower = meter_name.lower()
    # Remove common suffixes
    for suffix in [
        "cached input tokens",
        "input tokens",
        "output tokens",
        "cache read input tokens",
    ]:
        if lower.endswith(suffix):
            return meter_name[: -len(suffix)].strip()
    return None


def build_model_pricing(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Build a model -> {input/output/cache_read: price_per_million} mapping."""
    by_model: dict[str, dict[str, float]] = defaultdict(dict)

    for item in items:
        meter_name = item.get("meterName", "")
        unit_price = item.get("unitPrice", 0.0)
        unit_of_measure = item.get("unitOfMeasure", "")

        # Only process token-based pricing
        if "Token" not in unit_of_measure:
            continue

        category = _classify_meter(meter_name)
        if category is None:
            continue

        model_name = _extract_model_name(meter_name)
        if model_name is None:
            continue

        # Convert to per-million tokens (API returns per 1K tokens)
        price_per_million = unit_price * 1000
        by_model[model_name][category] = price_per_million

    return dict(by_model)


def _format_price(price: float) -> str:
    """Format a per-million-token price, dropping unnecessary trailing zeros."""
    if price == 0.0:
        return "0.0"
    formatted = f"{price:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted += ".0"
    return formatted


def _model_sort_key(name: str) -> str:
    """Normalize model name for sorting."""
    return name.lower().replace("-", " ")


def generate_pricing_code(pricing: dict[str, dict[str, float]]) -> str:
    """Generate the _PRICING dict entries as Python code."""
    entries: list[str] = []

    for model_name in sorted(pricing.keys(), key=_model_sort_key):
        prices = pricing[model_name]
        input_price = prices.get("input")
        output_price = prices.get("output")
        cache_read_price = prices.get("cache_read")

        if input_price is None and output_price is None:
            continue

        # Derive a clean model key (lowercase, hyphens)
        model_key = model_name.lower().replace(" ", "-")

        lines = [f'    "{model_key}": ModelPricing(']
        lines.append("        tiers=[")
        lines.append("            PricingTier(")

        if input_price is not None:
            lines.append(f"                input_cost_per_token=per_million_tokens({_format_price(input_price)}),")
        else:
            lines.append("                input_cost_per_token=0.0,")

        if output_price is not None:
            lines.append(f"                output_cost_per_token=per_million_tokens({_format_price(output_price)}),")
        else:
            lines.append("                output_cost_per_token=0.0,")

        if cache_read_price is not None and cache_read_price > 0:
            lines.append(
                f"                cache_read_cost_per_token=per_million_tokens({_format_price(cache_read_price)}),"
            )

        lines.append("            )")
        lines.append("        ]")
        lines.append("    ),")
        entries.append("\n".join(lines))

    return "\n".join(entries)


def generate_cost_py(entries_code: str) -> str:
    """Generate the full cost.py file content."""
    return f'''\
"""Azure AI Foundry pricing data and cost calculation.

Prices are for Global Standard (pay-as-you-go) deployments.
Use ``register_pricing()`` on ``AzureFoundryProvider`` for provisioned
deployments, regional overrides, or models not listed here.

Auto-generated by scripts/update_azure_foundry_pricing.py — manual edits will be overwritten.
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {{
{entries_code}
}}

# Pre-sorted by key length descending for longest-prefix matching
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_azure_foundry_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an Azure AI Foundry API call. Returns None if model pricing is unknown."""
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


def run(*, write: bool) -> None:
    """Main entry point."""
    print("Fetching Azure OpenAI pricing from Retail Prices API...", file=sys.stderr)
    items = fetch_prices()
    print(f"  Fetched {len(items)} pricing entries", file=sys.stderr)

    pricing = build_model_pricing(items)
    print(f"  Found pricing for {len(pricing)} models:", file=sys.stderr)
    for name in sorted(pricing.keys(), key=_model_sort_key):
        prices = pricing[name]
        parts = [f"{k}=${v:.4f}/M" for k, v in sorted(prices.items())]
        print(f"    {name}: {', '.join(parts)}", file=sys.stderr)

    entries_code = generate_pricing_code(pricing)
    output = generate_cost_py(entries_code)

    if write:
        COST_PY.write_text(output)
        print(f"\nWrote {COST_PY}", file=sys.stderr)
    else:
        print(output)
        print("\nPass --write to update cost.py in-place", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Azure AI Foundry pricing data")
    parser.add_argument("--write", action="store_true", help="Write directly to cost.py")
    args = parser.parse_args()
    run(write=args.write)
