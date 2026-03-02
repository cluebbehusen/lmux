#!/usr/bin/env python3
"""Generate the Vertex AI pricing dict for cost.py.

Usage:
    python scripts/update_vertex_pricing.py          # print to stdout
    python scripts/update_vertex_pricing.py --write   # update cost.py in-place

To update pricing:
    1. Edit the MODELS list below with current prices from
       https://cloud.google.com/vertex-ai/generative-ai/pricing
    2. Run with --write to regenerate cost.py

Prices are per million tokens (standard on-demand, global endpoint).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

COST_PY = Path(__file__).resolve().parent.parent / "packages/lmux-gcp-vertex/src/lmux_gcp_vertex/cost.py"


@dataclass
class Tier:
    input: float
    output: float
    cache_read: float | None = None
    cache_creation: float | None = None
    min_input_tokens: int = 0


@dataclass
class Model:
    prefix: str
    tiers: list[Tier] = field(default_factory=list)


@dataclass
class ModelGroup:
    heading: str
    models: list[Model]


# ────────────────────────────────────────────────────────────────────────────
# Pricing data — update this when prices change.
#
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
#         https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude
#         https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-mistral
#         https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-llama
# ────────────────────────────────────────────────────────────────────────────

MODELS: list[ModelGroup] = [
    ModelGroup(
        heading="Google Gemini 3 (Preview)",
        models=[
            Model(
                "gemini-3.1-pro-preview",
                [
                    Tier(input=2.00, output=12.00, cache_read=0.20),
                    Tier(input=4.00, output=18.00, cache_read=0.40, min_input_tokens=200_000),
                ],
            ),
            Model(
                "gemini-3-pro-preview",
                [
                    Tier(input=2.00, output=12.00, cache_read=0.20),
                    Tier(input=4.00, output=18.00, cache_read=0.40, min_input_tokens=200_000),
                ],
            ),
            Model(
                "gemini-3-flash-preview",
                [
                    Tier(input=0.50, output=3.00, cache_read=0.05),
                ],
            ),
        ],
    ),
    ModelGroup(
        heading="Google Gemini 2.5",
        models=[
            Model(
                "gemini-2.5-pro",
                [
                    Tier(input=1.25, output=10.00, cache_read=0.125),
                    Tier(input=2.50, output=15.00, cache_read=0.250, min_input_tokens=200_000),
                ],
            ),
            Model(
                "gemini-2.5-flash",
                [
                    Tier(input=0.30, output=2.50, cache_read=0.030),
                ],
            ),
            Model(
                "gemini-2.5-flash-lite",
                [
                    Tier(input=0.10, output=0.40, cache_read=0.010),
                ],
            ),
        ],
    ),
    ModelGroup(
        heading="Google Gemini 2.0",
        models=[
            Model("gemini-2.0-flash", [Tier(input=0.15, output=0.60)]),
            Model("gemini-2.0-flash-lite", [Tier(input=0.075, output=0.30)]),
        ],
    ),
    ModelGroup(
        heading="Google Gemini 1.5",
        models=[
            Model(
                "gemini-1.5-pro",
                [
                    Tier(input=1.25, output=5.00, cache_read=0.3125),
                    Tier(input=2.50, output=10.00, cache_read=0.625, min_input_tokens=128_000),
                ],
            ),
            Model(
                "gemini-1.5-flash",
                [
                    Tier(input=0.075, output=0.30, cache_read=0.01875),
                    Tier(input=0.15, output=0.60, cache_read=0.0375, min_input_tokens=128_000),
                ],
            ),
        ],
    ),
    ModelGroup(
        heading="Anthropic Claude (via Vertex AI)",
        models=[
            Model("claude-opus-4-6", [Tier(input=5.00, output=25.00, cache_read=0.50)]),
            Model("claude-sonnet-4-6", [Tier(input=3.00, output=15.00, cache_read=0.30)]),
            Model("claude-opus-4-5", [Tier(input=5.00, output=25.00, cache_read=0.50)]),
            Model("claude-sonnet-4-5", [Tier(input=3.00, output=15.00, cache_read=0.30)]),
            Model("claude-opus-4-1", [Tier(input=15.00, output=75.00, cache_read=1.50)]),
            Model("claude-opus-4", [Tier(input=15.00, output=75.00, cache_read=1.50)]),
            Model("claude-sonnet-4", [Tier(input=3.00, output=15.00, cache_read=0.30)]),
            Model("claude-haiku-4-5", [Tier(input=1.00, output=5.00, cache_read=0.10)]),
            Model("claude-3-5-sonnet-v2", [Tier(input=3.00, output=15.00, cache_read=0.30)]),
            Model("claude-3-5-haiku", [Tier(input=0.80, output=4.00, cache_read=0.08)]),
            Model("claude-3-opus", [Tier(input=15.00, output=75.00, cache_read=1.50)]),
            Model("claude-3-haiku", [Tier(input=0.25, output=1.25, cache_read=0.03)]),
        ],
    ),
    ModelGroup(
        heading="Mistral (via Vertex AI)",
        models=[
            Model("mistral-medium-3", [Tier(input=1.00, output=3.00)]),
            Model("mistral-small-2503", [Tier(input=0.20, output=0.60)]),
            Model("codestral-2", [Tier(input=0.50, output=1.50)]),
        ],
    ),
    ModelGroup(
        heading="Meta Llama (via Vertex AI)",
        models=[
            Model("llama-4-maverick-17b-128e-instruct-maas", [Tier(input=0.50, output=0.77)]),
            Model("llama-4-scout-17b-16e-instruct-maas", [Tier(input=0.11, output=0.34)]),
            Model("llama-3.3-70b-instruct-maas", [Tier(input=0.20, output=0.20)]),
        ],
    ),
    ModelGroup(
        heading="Embedding models",
        models=[
            Model("text-embedding-005", [Tier(input=0.025, output=0)]),
            Model("text-embedding-004", [Tier(input=0.025, output=0)]),
            Model("text-multilingual-embedding-002", [Tier(input=0.025, output=0)]),
        ],
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Code generation
# ────────────────────────────────────────────────────────────────────────────

HEADER = '''\
"""GCP Vertex AI pricing data and cost calculation.

Prices are for standard on-demand (global endpoint) pricing.
Auto-generated by scripts/update_vertex_pricing.py — manual edits will be overwritten.
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
'''

FOOTER = '''\
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
'''


def _format_price(price: float) -> str:
    """Format a per-million-token price, dropping unnecessary trailing zeros."""
    if price == 0:
        return "0.0"
    formatted = f"{price:.6f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted += ".0"
    return formatted


def _generate_tier(tier: Tier, indent: str) -> str:
    """Generate a single PricingTier(...) block."""
    lines = [f"{indent}PricingTier("]
    inner = indent + "    "
    lines.append(f"{inner}input_cost_per_token=per_million_tokens({_format_price(tier.input)}),")

    if tier.output == 0:
        lines.append(f"{inner}output_cost_per_token=0.0,")
    else:
        lines.append(f"{inner}output_cost_per_token=per_million_tokens({_format_price(tier.output)}),")

    if tier.cache_read is not None and tier.cache_read > 0:
        lines.append(f"{inner}cache_read_cost_per_token=per_million_tokens({_format_price(tier.cache_read)}),")
    if tier.cache_creation is not None and tier.cache_creation > 0:
        lines.append(f"{inner}cache_creation_cost_per_token=per_million_tokens({_format_price(tier.cache_creation)}),")
    if tier.min_input_tokens > 0:
        lines.append(f"{inner}min_input_tokens={tier.min_input_tokens:_},")

    lines.append(f"{indent}),")
    return "\n".join(lines)


def _generate_model(model: Model) -> str:
    """Generate a single _PRICING entry."""
    tier_lines = [_generate_tier(tier, "            ") for tier in model.tiers]
    tiers_body = "\n".join(tier_lines)
    return f'    "{model.prefix}": ModelPricing(\n        tiers=[\n{tiers_body}\n        ],\n    ),'


def generate_cost_py() -> str:
    """Generate the full cost.py file content."""
    sections: list[str] = []

    for group in MODELS:
        pad = max(1, 55 - len(group.heading))
        sections.append(f"    # ── {group.heading} " + "─" * pad)
        sections.extend(_generate_model(model) for model in group.models)

    return HEADER + "\n".join(sections) + "\n" + FOOTER


def run(*, write: bool) -> None:
    output = generate_cost_py()

    if write:
        COST_PY.write_text(output)
        print(f"Wrote {COST_PY}", file=sys.stderr)
    else:
        print(output)
        print(f"\nPass --write to update {COST_PY.name} in-place", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vertex AI pricing data for cost.py")
    parser.add_argument("--write", action="store_true", help="Write directly to cost.py")
    args = parser.parse_args()
    run(write=args.write)
