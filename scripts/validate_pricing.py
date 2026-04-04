"""Cross-reference lmux pricing against external sources.

Compares base-tier per-million-token prices from every lmux provider package
against three independent pricing databases:

  1. LiteLLM  — model_prices_and_context_window.json (GitHub)
  2. OpenRouter — /api/v1/models (live API)
  3. genai-prices (pydantic) — data.json (GitHub)

Usage::

    python scripts/validate_pricing.py               # default 1 % tolerance
    python scripts/validate_pricing.py --tolerance 5  # 5 % tolerance
    python scripts/validate_pricing.py --provider openai  # single provider
"""

import argparse
import importlib
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MILLION = 1_000_000


@dataclass
class PricePoint:
    """Per-million-token prices for a single model (base tier only)."""

    input: Decimal
    output: Decimal
    cache_read: Decimal | None = None
    cache_write: Decimal | None = None


@dataclass
class TieredPricing:
    """All pricing tiers for a model: list of (min_input_tokens, PricePoint)."""

    tiers: list[tuple[int, PricePoint]]


@dataclass
class Mismatch:
    model: str
    field: str
    lmux_value: Decimal
    external_value: Decimal
    pct_diff: Decimal


@dataclass
class SourceReport:
    source_name: str
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    missing_from_source: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# lmux pricing extraction
# ---------------------------------------------------------------------------


@dataclass
class ProviderSpec:
    module: str
    name: str
    litellm_prefixes: list[str]
    genai_provider: str | None
    openrouter_prefixes: list[str]


PROVIDER_SPECS: list[ProviderSpec] = [
    ProviderSpec("lmux_openai.cost", "openai", ["openai"], "openai", ["openai/"]),
    ProviderSpec("lmux_anthropic.cost", "anthropic", ["anthropic", ""], "anthropic", ["anthropic/"]),
    ProviderSpec("lmux_groq.cost", "groq", ["groq/", ""], "groq", ["groq/"]),
    ProviderSpec(
        "lmux_gcp_vertex.cost",
        "gcp-vertex",
        ["vertex_ai/", "vertex_ai-chat-models/", "vertex_ai-language-models/", "gemini/", ""],
        "google",
        ["google/"],
    ),
    ProviderSpec(
        "lmux_aws_bedrock.cost",
        "aws-bedrock",
        ["bedrock/", ""],
        "aws",
        [],
    ),
    ProviderSpec(
        "lmux_azure_foundry.cost",
        "azure-foundry",
        ["azure/", "azure/global/", "azure/global-standard/", "azure_ai/", "azure_ai/global/", ""],
        "azure",
        ["azure/", "microsoft/"],
    ),
]


def _to_per_million(per_token: float) -> Decimal:
    return Decimal(str(per_token)) * MILLION


def extract_lmux_pricing(module_name: str) -> dict[str, PricePoint]:
    """Import a provider's cost module and extract base-tier pricing."""
    mod = importlib.import_module(module_name)
    pricing_dict: dict[str, Any] = mod._PRICING
    result: dict[str, PricePoint] = {}
    for model_id, model_pricing in pricing_dict.items():
        base_tier = model_pricing.tiers[0]
        result[model_id] = PricePoint(
            input=_to_per_million(base_tier.input_cost_per_token),
            output=_to_per_million(base_tier.output_cost_per_token),
            cache_read=_to_per_million(base_tier.cache_read_cost_per_token)
            if base_tier.cache_read_cost_per_token is not None
            else None,
            cache_write=_to_per_million(base_tier.cache_creation_cost_per_token)
            if base_tier.cache_creation_cost_per_token is not None
            else None,
        )
    return result


def extract_lmux_tiered_pricing(module_name: str) -> dict[str, TieredPricing]:
    """Import a provider's cost module and extract all pricing tiers."""
    mod = importlib.import_module(module_name)
    pricing_dict: dict[str, Any] = mod._PRICING
    result: dict[str, TieredPricing] = {}
    for model_id, model_pricing in pricing_dict.items():
        if len(model_pricing.tiers) <= 1:
            continue
        result[model_id] = TieredPricing(
            tiers=[
                (
                    tier.min_input_tokens,
                    PricePoint(
                        input=_to_per_million(tier.input_cost_per_token),
                        output=_to_per_million(tier.output_cost_per_token),
                        cache_read=_to_per_million(tier.cache_read_cost_per_token)
                        if tier.cache_read_cost_per_token is not None
                        else None,
                        cache_write=_to_per_million(tier.cache_creation_cost_per_token)
                        if tier.cache_creation_cost_per_token is not None
                        else None,
                    ),
                )
                for tier in model_pricing.tiers
            ]
        )
    return result


# ---------------------------------------------------------------------------
# External source: LiteLLM
# ---------------------------------------------------------------------------

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def fetch_litellm() -> dict[str, Any]:
    """Fetch and parse the LiteLLM pricing JSON."""
    print("  Fetching LiteLLM pricing database...")  # noqa: T201
    req = urllib.request.Request(LITELLM_URL, headers={"User-Agent": "lmux-pricing-validator/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        result: dict[str, Any] = json.loads(resp.read().decode())
        return result


def _litellm_entry_to_price(entry: dict[str, object]) -> PricePoint | None:
    """Convert a LiteLLM entry to a PricePoint, or None if missing required fields."""
    input_cost = entry.get("input_cost_per_token")
    output_cost = entry.get("output_cost_per_token")
    if input_cost is None or output_cost is None:
        return None
    cache_read = entry.get("cache_read_input_token_cost")
    cache_write = entry.get("cache_creation_input_token_cost")
    return PricePoint(
        input=Decimal(str(input_cost)) * MILLION,
        output=Decimal(str(output_cost)) * MILLION,
        cache_read=Decimal(str(cache_read)) * MILLION if cache_read is not None else None,
        cache_write=Decimal(str(cache_write)) * MILLION if cache_write is not None else None,
    )


_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+$")


def _litellm_build_candidates(model_id: str, provider_prefixes: list[str]) -> list[str]:
    """Build candidate keys to try when looking up a model in LiteLLM."""
    base_ids = [model_id]
    # Strip Bedrock version suffix (e.g. -v1) to match date-versioned LiteLLM keys
    # e.g. anthropic.claude-sonnet-4-v1 → anthropic.claude-sonnet-4
    stripped = _BEDROCK_VERSION_SUFFIX.sub("", model_id)
    if stripped != model_id:
        base_ids.append(stripped)

    candidates: list[str] = []
    for mid in base_ids:
        for pfx in provider_prefixes:
            sep = "" if pfx.endswith("/") or pfx == "" else "/"
            candidates.append(f"{pfx}{sep}{mid}")
        if mid not in candidates:
            candidates.append(mid)
    # Also try with :0 suffix (common in Bedrock model IDs)
    candidates += [c + ":0" for c in candidates if not c.endswith(":0")]
    return candidates


def _litellm_find_entry(
    data: dict[str, Any],
    model_id: str,
    provider_prefixes: list[str],
) -> dict[str, object] | None:
    """Find the raw LiteLLM entry for a model, trying prefixed/bare/case-insensitive."""
    candidates = _litellm_build_candidates(model_id, provider_prefixes)

    # Exact match
    for candidate in candidates:
        if candidate in data:
            return data[candidate]

    # Case-insensitive fallback
    # Build model ID variants to try as prefixes (original + stripped version suffix)
    model_variants = [model_id.lower()]
    stripped = _BEDROCK_VERSION_SUFFIX.sub("", model_id).lower()
    if stripped != model_variants[0]:
        model_variants.append(stripped)

    for key, entry in data.items():
        key_lower = key.lower()
        for candidate in candidates:
            if key_lower == candidate.lower():
                return entry
        for pfx in provider_prefixes:
            sep = "" if pfx.endswith("/") or pfx == "" else "/"
            full_prefix = f"{pfx}{sep}".lower()
            key_suffix = key_lower[len(full_prefix) :]
            if not key_lower.startswith(full_prefix):
                continue
            for variant in model_variants:
                if not key_suffix.startswith(variant):
                    continue
                # For stripped variants, ensure the remainder is a date/version separator
                # (e.g. -20250514) not another model component (e.g. -6 in sonnet-4-6)
                remainder = key_suffix[len(variant) :]
                is_stripped = variant != model_variants[0]
                if is_stripped and remainder and not _is_date_suffix(remainder):
                    continue
                return entry

    return None


_DATE_SUFFIX = re.compile(r"^(-\d{8}|-v\d|:)")


def _is_date_suffix(remainder: str) -> bool:
    """Check if a key remainder starts with a date-like pattern (-YYYYMMDD or -v or :)."""
    return bool(_DATE_SUFFIX.match(remainder))


def litellm_lookup(
    data: dict[str, Any],
    model_id: str,
    provider_prefixes: list[str],
) -> PricePoint | None:
    """Look up a model in LiteLLM data, trying provider-prefixed and bare names."""
    entry = _litellm_find_entry(data, model_id, provider_prefixes)
    if entry is None:
        return None
    return _litellm_entry_to_price(entry)


_ABOVE_PATTERN = re.compile(r"_above_(\d+)k_tokens$")


def _litellm_extract_tiers(entry: dict[str, object]) -> TieredPricing | None:
    """Extract tiered pricing from a LiteLLM entry's _above_*k_tokens fields."""
    base = _litellm_entry_to_price(entry)
    if base is None:
        return None

    # Collect all _above_ thresholds present
    thresholds: set[int] = set()
    for key in entry:
        m = _ABOVE_PATTERN.search(key)
        if m:
            thresholds.add(int(m.group(1)) * 1000)

    if not thresholds:
        return None

    tiers: list[tuple[int, PricePoint]] = [(0, base)]
    for threshold in sorted(thresholds):
        suffix = f"_above_{threshold // 1000}k_tokens"
        input_cost = entry.get(f"input_cost_per_token{suffix}")
        output_cost = entry.get(f"output_cost_per_token{suffix}")
        if input_cost is None or output_cost is None:
            continue
        cache_read = entry.get(f"cache_read_input_token_cost{suffix}")
        cache_write = entry.get(f"cache_creation_input_token_cost{suffix}")
        tiers.append(
            (
                threshold,
                PricePoint(
                    input=Decimal(str(input_cost)) * MILLION,
                    output=Decimal(str(output_cost)) * MILLION,
                    cache_read=Decimal(str(cache_read)) * MILLION if cache_read is not None else None,
                    cache_write=Decimal(str(cache_write)) * MILLION if cache_write is not None else None,
                ),
            )
        )

    return TieredPricing(tiers=tiers) if len(tiers) > 1 else None


# ---------------------------------------------------------------------------
# External source: OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter() -> dict[str, Any]:
    """Fetch and index OpenRouter models by their base model name."""
    print("  Fetching OpenRouter models API...")  # noqa: T201
    req = urllib.request.Request(OPENROUTER_URL, headers={"User-Agent": "lmux-pricing-validator/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw = json.loads(resp.read().decode())
    result: dict[str, Any] = {}
    for model in raw.get("data", []):
        model_id = model.get("id", "")
        result[model_id] = model
    return result


def openrouter_lookup(
    data: dict[str, Any],
    model_id: str,
    provider_prefixes: list[str],
) -> PricePoint | None:
    """Look up a model in OpenRouter data."""
    candidates = [f"{pfx}{model_id}" for pfx in provider_prefixes] + [model_id]
    for candidate in candidates:
        if candidate in data:
            pricing = data[candidate].get("pricing", {})
            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")
            if prompt_price is None or completion_price is None:
                continue
            try:
                input_per_token = Decimal(str(prompt_price))
                output_per_token = Decimal(str(completion_price))
            except (ValueError, ArithmeticError):
                continue
            if input_per_token == 0 and output_per_token == 0:
                continue
            cache_read_raw = pricing.get("input_cache_read")
            cache_write_raw = pricing.get("input_cache_write")
            return PricePoint(
                input=input_per_token * MILLION,
                output=output_per_token * MILLION,
                cache_read=Decimal(str(cache_read_raw)) * MILLION
                if cache_read_raw is not None and str(cache_read_raw) != "0"
                else None,
                cache_write=Decimal(str(cache_write_raw)) * MILLION
                if cache_write_raw is not None and str(cache_write_raw) != "0"
                else None,
            )
    return None


# ---------------------------------------------------------------------------
# External source: genai-prices (pydantic)
# ---------------------------------------------------------------------------

GENAI_PRICES_URL = "https://raw.githubusercontent.com/pydantic/genai-prices/main/prices/data.json"


def fetch_genai_prices() -> dict[str, Any]:
    """Fetch genai-prices data and index by provider ID."""
    print("  Fetching genai-prices database...")  # noqa: T201
    req = urllib.request.Request(GENAI_PRICES_URL, headers={"User-Agent": "lmux-pricing-validator/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw = json.loads(resp.read().decode())
    # raw is a list of provider objects; index by provider ID
    result: dict[str, Any] = {}
    for provider in raw:
        pid = provider.get("id", "")
        result[pid] = provider
    return result


def _extract_genai_base_price(value: object) -> Decimal | None:
    """Extract base price from a genai-prices field (scalar or tiered)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, dict):
        base: object = value.get("base")  # pyright: ignore[reportUnknownVariableType]
        if base is not None:
            return Decimal(str(base))  # pyright: ignore[reportUnknownArgumentType]
    return None


def _resolve_genai_prices(prices_data: object) -> dict[str, object] | None:
    """Resolve genai-prices data to a flat dict, handling conditional price lists."""
    target: object = prices_data
    if isinstance(target, list):
        if not target:
            return None
        last_entry: object = target[-1]  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(last_entry, dict):
            return None
        target = last_entry.get("prices", last_entry)  # pyright: ignore[reportUnknownVariableType]
    if isinstance(target, dict):
        return dict(target)  # pyright: ignore[reportUnknownArgumentType]
    return None


def _extract_genai_model(model: dict[str, object]) -> PricePoint | None:
    """Extract a PricePoint from a genai-prices model entry."""
    resolved = _resolve_genai_prices(model.get("prices"))
    if resolved is None:
        return None

    input_price = _extract_genai_base_price(resolved.get("input_mtok"))
    output_price = _extract_genai_base_price(resolved.get("output_mtok"))
    if input_price is None or output_price is None:
        return None

    return PricePoint(
        input=input_price,
        output=output_price,
        cache_read=_extract_genai_base_price(resolved.get("cache_read_mtok")),
        cache_write=_extract_genai_base_price(resolved.get("cache_write_mtok")),
    )


def genai_prices_lookup(
    data: dict[str, Any],
    model_id: str,
    provider_name: str | None,
) -> PricePoint | None:
    """Look up a model in genai-prices data."""
    if provider_name is None:
        return None

    provider = data.get(provider_name)
    if provider is None:
        return None

    for model in provider.get("models", []):
        genai_id = model.get("id", "")
        if genai_id != model_id:
            match_rules = model.get("match", {})
            if not _genai_match(match_rules, model_id):
                continue
        result = _extract_genai_model(model)
        if result is not None:
            return result
    return None


def _genai_match(match_rules: dict[str, Any], model_id: str) -> bool:
    """Evaluate genai-prices match rules against a model ID."""
    if "equals" in match_rules:
        return model_id == match_rules["equals"]
    if "starts_with" in match_rules:
        return model_id.startswith(match_rules["starts_with"])
    if "or" in match_rules:
        return any(_genai_match(rule, model_id) for rule in match_rules["or"])
    if "and" in match_rules:
        return all(_genai_match(rule, model_id) for rule in match_rules["and"])
    return False


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_prices(
    lmux_price: PricePoint,
    external_price: PricePoint,
    model_id: str,
    tolerance_pct: Decimal,
) -> list[Mismatch]:
    """Compare two price points and return mismatches exceeding tolerance."""
    mismatches: list[Mismatch] = []
    fields = [
        ("input", lmux_price.input, external_price.input),
        ("output", lmux_price.output, external_price.output),
    ]
    if lmux_price.cache_read is not None and external_price.cache_read is not None:
        fields.append(("cache_read", lmux_price.cache_read, external_price.cache_read))
    if lmux_price.cache_write is not None and external_price.cache_write is not None:
        fields.append(("cache_write", lmux_price.cache_write, external_price.cache_write))

    for field_name, lmux_val, ext_val in fields:
        if ext_val == 0 and lmux_val == 0:
            continue
        pct = Decimal(100) if ext_val == 0 else abs((lmux_val - ext_val) / ext_val) * 100
        if pct > tolerance_pct:
            mismatches.append(
                Mismatch(
                    model=model_id,
                    field=field_name,
                    lmux_value=lmux_val,
                    external_value=ext_val,
                    pct_diff=pct,
                )
            )
    return mismatches


def _litellm_lookup_tiered(
    data: dict[str, Any],
    lmux_tiered: dict[str, TieredPricing],
    provider_prefixes: list[str],
) -> dict[str, TieredPricing]:
    """Look up tiered pricing from LiteLLM for all models that have tiers in lmux."""
    result: dict[str, TieredPricing] = {}
    for model_id in lmux_tiered:
        entry = _litellm_find_entry(data, model_id, provider_prefixes)
        if entry is None:
            continue
        tiered = _litellm_extract_tiers(entry)
        if tiered is not None:
            result[model_id] = tiered
    return result


def compare_tiered_prices(
    lmux_tiered: dict[str, TieredPricing],
    external_tiered: dict[str, TieredPricing],
    tolerance_pct: Decimal,
) -> SourceReport:
    """Compare tiered pricing between lmux and an external source."""
    report = SourceReport(source_name="LiteLLM (tiered)")
    for model_id, lmux_tp in lmux_tiered.items():
        ext_tp = external_tiered.get(model_id)
        if ext_tp is None:
            report.missing_from_source.append(model_id)
            continue
        report.matched += 1
        # Compare tier by tier — match on threshold
        ext_by_threshold = dict(ext_tp.tiers)
        for threshold, lmux_pp in lmux_tp.tiers:
            if threshold == 0:
                continue  # base tier already compared separately
            ext_pp = ext_by_threshold.get(threshold)
            if ext_pp is None:
                report.missing_from_source.append(f"{model_id} (>{threshold:,} tier)")
                continue
            label = f"{model_id} (>{threshold:,})"
            report.mismatches.extend(compare_prices(lmux_pp, ext_pp, label, tolerance_pct))
    return report


# ---------------------------------------------------------------------------
# Calculated cost comparison
# ---------------------------------------------------------------------------


def _find_provider_calc_fn(module_name: str) -> Callable[[str, object], object] | None:
    """Find the provider-specific calculate function (e.g. calculate_openai_cost)."""
    mod = importlib.import_module(module_name)
    for name in dir(mod):
        is_provider_calc = name.startswith("calculate_") and name.endswith("_cost") and name != "calculate_cost"
        if is_provider_calc and callable(getattr(mod, name)):
            return getattr(mod, name)
    return None


def _resolve_ext_price(
    model_id: str,
    input_tokens: int,
    base_price: PricePoint,
    external_tiered: dict[str, TieredPricing] | None,
) -> PricePoint:
    """Pick the right external PricePoint for a given input token count, considering tiers."""
    if external_tiered is None:
        return base_price
    tiered = external_tiered.get(model_id)
    if tiered is None:
        return base_price
    # Pick highest tier whose threshold is exceeded (same logic as lmux's _resolve_tier)
    best = base_price
    for threshold, pp in tiered.tiers:
        if input_tokens > threshold:
            best = pp
    return best


def compare_calculated_costs(
    module_name: str,
    lmux_prices: dict[str, PricePoint],
    external_prices: dict[str, PricePoint],
    tolerance_pct: Decimal,
    external_tiered: dict[str, TieredPricing] | None = None,
) -> list[Mismatch]:
    """Compare lmux's calculate_*_cost() output against external prices for sample usage scenarios."""
    calc_fn = _find_provider_calc_fn(module_name)
    if calc_fn is None:
        return []

    from lmux.types import Usage  # noqa: PLC0415

    mismatches: list[Mismatch] = []
    scenarios = [
        (1000, 500, 0),
        (100_000, 50_000, 10_000),
        (1_000_000, 200_000, 0),
    ]

    for model_id in lmux_prices:
        if model_id not in external_prices:
            continue

        for in_tok, out_tok, cache_tok in scenarios:
            usage = Usage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                cache_read_tokens=cache_tok if cache_tok > 0 else None,
            )
            lmux_cost = calc_fn(model_id, usage)
            if lmux_cost is None:
                continue

            ext = _resolve_ext_price(model_id, in_tok, external_prices[model_id], external_tiered)
            billable_input = in_tok - cache_tok
            expected_input_cost = float(ext.input) / MILLION * billable_input
            expected_output_cost = float(ext.output) / MILLION * out_tok
            expected_cache_cost = float(ext.cache_read or 0) / MILLION * cache_tok if cache_tok > 0 else 0
            expected_total = expected_input_cost + expected_output_cost + expected_cache_cost
            if expected_total == 0:
                continue

            actual_total: float = lmux_cost.total_cost  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
            pct = abs(Decimal(str(actual_total)) - Decimal(str(expected_total))) / Decimal(str(expected_total)) * 100  # pyright: ignore[reportUnknownArgumentType]
            if pct <= tolerance_pct:
                continue

            mismatches.append(
                Mismatch(
                    model=f"{model_id} ({in_tok:,}in/{out_tok:,}out/{cache_tok:,}cache)",
                    field="total_cost",
                    lmux_value=Decimal(str(round(actual_total, 8))),  # pyright: ignore[reportUnknownArgumentType]
                    external_value=Decimal(str(round(expected_total, 8))),
                    pct_diff=pct,
                )
            )
    return mismatches


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print(msg: str = "") -> None:
    """Print a line to stdout."""
    print(msg)  # noqa: T201


def print_report(
    provider_name: str,
    reports: list[SourceReport],
    calc_mismatches: dict[str, list[Mismatch]],
) -> bool:
    """Print comparison report. Returns True if any mismatches found."""
    has_issues = False
    _print(f"\n{'=' * 70}")
    _print(f"  {provider_name.upper()}")
    _print(f"{'=' * 70}")

    for report in reports:
        total_models = report.matched + len(report.missing_from_source)
        _print(f"\n  vs {report.source_name}: {report.matched}/{total_models} models matched")

        if report.mismatches:
            has_issues = True
            _print(f"  {'!' * 3} PRICE MISMATCHES {'!' * 3}")
            for m in report.mismatches:
                lmux_s = f"lmux={m.lmux_value:>10.4f}"
                ext_s = f"ext={m.external_value:>10.4f}"
                _print(f"    {m.model:40s} {m.field:12s}  {lmux_s}  {ext_s}  diff={m.pct_diff:>6.2f}%")

        if report.missing_from_source:
            _print(f"  Not found in {report.source_name}:")
            for model in report.missing_from_source:
                _print(f"    - {model}")

    for source_name, mismatches in calc_mismatches.items():
        if mismatches:
            has_issues = True
            _print(f"\n  {'!' * 3} CALCULATED COST MISMATCHES vs {source_name} {'!' * 3}")
            for m in mismatches:
                lmux_s = f"lmux=${m.lmux_value:>12.8f}"
                ext_s = f"ext=${m.external_value:>12.8f}"
                _print(f"    {m.model:55s}  {lmux_s}  {ext_s}  diff={m.pct_diff:>6.2f}%")

    if not has_issues:
        _print("\n  All prices verified against external sources.")

    return has_issues


# ---------------------------------------------------------------------------
# External data fetching
# ---------------------------------------------------------------------------


@dataclass
class ExternalData:
    litellm: dict[str, Any] | None = None
    openrouter: dict[str, Any] | None = None
    genai: dict[str, Any] | None = None


def _fetch_all(args: argparse.Namespace) -> ExternalData:
    """Fetch all enabled external pricing sources."""
    _print("Fetching external pricing data...")
    data = ExternalData()

    if not args.skip_litellm:
        try:
            data.litellm = fetch_litellm()
            _print(f"  LiteLLM: {len(data.litellm)} models loaded")
        except Exception as e:  # noqa: BLE001
            _print(f"  LiteLLM: FAILED ({e})")

    if not args.skip_openrouter:
        try:
            data.openrouter = fetch_openrouter()
            _print(f"  OpenRouter: {len(data.openrouter)} models loaded")
        except Exception as e:  # noqa: BLE001
            _print(f"  OpenRouter: FAILED ({e})")

    if not args.skip_genai_prices:
        try:
            data.genai = fetch_genai_prices()
            total_models = sum(len(p.get("models", [])) for p in data.genai.values())
            _print(f"  genai-prices: {total_models} models across {len(data.genai)} providers")
        except Exception as e:  # noqa: BLE001
            _print(f"  genai-prices: FAILED ({e})")

    return data


# ---------------------------------------------------------------------------
# Per-source comparison
# ---------------------------------------------------------------------------


def _compare_against_source(
    source_name: str,
    source_data: dict[str, PricePoint],
    lmux_prices: dict[str, PricePoint],
    tolerance: Decimal,
) -> SourceReport:
    """Compare lmux prices against pre-looked-up external prices."""
    report = SourceReport(source_name=source_name)
    for model_id, lmux_price in lmux_prices.items():
        ext = source_data.get(model_id)
        if ext is None:
            report.missing_from_source.append(model_id)
            continue
        report.matched += 1
        report.mismatches.extend(compare_prices(lmux_price, ext, model_id, tolerance))
    return report


def _lookup_all(
    lookup_fn: Callable[..., PricePoint | None],
    source_data: dict[str, object],
    model_ids: list[str],
    lookup_key: str | list[str] | None,
) -> dict[str, PricePoint]:
    """Run a lookup function for every model and collect successful matches."""
    result: dict[str, PricePoint] = {}
    for model_id in model_ids:
        ext = lookup_fn(source_data, model_id, lookup_key)
        if ext is not None:
            result[model_id] = ext
    return result


def _validate_provider(
    spec: ProviderSpec,
    ext_data: ExternalData,
    tolerance: Decimal,
    *,
    skip_calculated: bool,
) -> bool:
    """Validate a single provider against all available sources. Returns True if issues found."""
    try:
        lmux_prices = extract_lmux_pricing(spec.module)
    except Exception as e:  # noqa: BLE001
        _print(f"\nWARNING: Could not load {spec.module}: {e}")
        return False

    model_ids = list(lmux_prices.keys())
    reports: list[SourceReport] = []
    calc_mismatches: dict[str, list[Mismatch]] = {}

    lmux_tiered = extract_lmux_tiered_pricing(spec.module)

    ext_tiered: dict[str, TieredPricing] | None = None
    if ext_data.litellm is not None:
        ext_prices = _lookup_all(litellm_lookup, ext_data.litellm, model_ids, spec.litellm_prefixes)
        reports.append(_compare_against_source("LiteLLM", ext_prices, lmux_prices, tolerance))
        # Tiered pricing comparison (LiteLLM only — OpenRouter/genai-prices lack tier data)
        if lmux_tiered:
            ext_tiered = _litellm_lookup_tiered(ext_data.litellm, lmux_tiered, spec.litellm_prefixes)
            reports.append(compare_tiered_prices(lmux_tiered, ext_tiered, tolerance))
        if not skip_calculated and ext_prices:
            calc_mismatches["LiteLLM"] = compare_calculated_costs(
                spec.module, lmux_prices, ext_prices, tolerance, ext_tiered
            )

    if ext_data.openrouter is not None:
        ext_prices = _lookup_all(openrouter_lookup, ext_data.openrouter, model_ids, spec.openrouter_prefixes)
        reports.append(_compare_against_source("OpenRouter", ext_prices, lmux_prices, tolerance))
        if not skip_calculated and ext_prices:
            calc_mismatches["OpenRouter"] = compare_calculated_costs(spec.module, lmux_prices, ext_prices, tolerance)

    if ext_data.genai is not None:
        ext_prices = _lookup_all(genai_prices_lookup, ext_data.genai, model_ids, spec.genai_provider)
        reports.append(_compare_against_source("genai-prices", ext_prices, lmux_prices, tolerance))
        if not skip_calculated and ext_prices:
            calc_mismatches["genai-prices"] = compare_calculated_costs(spec.module, lmux_prices, ext_prices, tolerance)

    return print_report(spec.name, reports, calc_mismatches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate lmux pricing against external sources")
    _ = parser.add_argument("--tolerance", type=float, default=1.0, help="Tolerance percentage (default: 1.0)")
    _ = parser.add_argument("--provider", type=str, default=None, help="Check a single provider (e.g. 'openai')")
    _ = parser.add_argument("--skip-litellm", action="store_true", help="Skip LiteLLM comparison")
    _ = parser.add_argument("--skip-openrouter", action="store_true", help="Skip OpenRouter comparison")
    _ = parser.add_argument("--skip-genai-prices", action="store_true", help="Skip genai-prices comparison")
    _ = parser.add_argument("--skip-calculated", action="store_true", help="Skip calculated cost comparison")
    args = parser.parse_args()

    tolerance = Decimal(str(args.tolerance))

    # Ensure packages are importable
    root = Path(__file__).resolve().parent.parent
    for pkg_dir in sorted(root.glob("packages/*/src")):
        src = str(pkg_dir)
        if src not in sys.path:
            sys.path.insert(0, src)

    ext_data = _fetch_all(args)
    if ext_data.litellm is None and ext_data.openrouter is None and ext_data.genai is None:
        _print("\nERROR: Could not fetch any external pricing data.")
        return 1

    any_issues = False
    for spec in PROVIDER_SPECS:
        if args.provider and args.provider != spec.name:
            continue
        if _validate_provider(spec, ext_data, tolerance, skip_calculated=args.skip_calculated):
            any_issues = True

    _print(f"\n{'=' * 70}")
    if any_issues:
        _print("  RESULT: Pricing discrepancies found (see above)")
        return 1
    _print("  RESULT: All prices match within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
