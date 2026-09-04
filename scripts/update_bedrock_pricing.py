#!/usr/bin/env python3
"""Generate AWS Bedrock cost.py from the AWS Pricing API and Bedrock API.

Fetches pricing from two unauthenticated API endpoints:
- AmazonBedrock: third-party models (DeepSeek, Gemma, Mistral, etc.)
- AmazonBedrockFoundationModels: Claude, Amazon Nova/Titan, Cohere, etc.

Then fetches real model and inference profile IDs from the Bedrock API
(via boto3's default credential chain) to ensure pricing keys match actual
Bedrock identifiers.

Regional overrides for every Region are included by default (a handful of Regions -- notably
GovCloud -- price some models above us-east-1); ``--regions`` narrows this for quick partial runs.

Usage:
    python3 scripts/update_bedrock_pricing.py --write            # all Regions
    python3 scripts/update_bedrock_pricing.py                    # stdout, all Regions
    python3 scripts/update_bedrock_pricing.py --regions eu-west-1 ap-northeast-1
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, fields, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, NoReturn

import boto3
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

API_BASE = "https://pricing.us-east-1.amazonaws.com"
DEFAULT_REGION = "us-east-1"
COST_PY_PATH = (
    Path(__file__).resolve().parent.parent / "packages" / "lmux-aws-bedrock" / "src" / "lmux_aws_bedrock" / "cost.py"
)
# The Anthropic-on-Bedrock subset is emitted here and shared with lmux-anthropic's
# native Bedrock provider (so Claude is priced identically by both). Everything else
# stays in COST_PY_PATH, which merges this back in.
SHARED_PRICING_PATH = (
    Path(__file__).resolve().parent.parent
    / "packages"
    / "lmux-bedrock-shared"
    / "src"
    / "lmux_bedrock_shared"
    / "pricing.py"
)

# Long-context tier threshold (tokens). Anthropic uses 200K for all models.
LCTX_THRESHOLD = 200_000

# ── Model ID mappings ────────────────────────────────────────────────────────

# Foundation Models API: servicename (after stripping " (Amazon Bedrock Edition)") -> Bedrock model ID
FM_SERVICENAME_MAP: dict[str, str] = {
    # Anthropic Claude. Retired models must be keyed by their real dated model ID. AWS drops a
    # model from list_foundation_models when it retires, so the catalog resolver can no longer
    # turn a dateless key into the ID a caller presents, and the dateless key would be written
    # out as an entry nothing can ever look up. See KNOWN_UNRESOLVED_IDS.
    "Claude Fable 5.1": "anthropic.claude-fable-5-1",
    "Claude Fable 5": "anthropic.claude-fable-5-v1",
    "Claude Mythos 5.1": "anthropic.claude-mythos-5-1",
    "Claude Mythos 5": "anthropic.claude-mythos-5-v1",
    "Claude Opus 5": "anthropic.claude-opus-5-v1",
    "Claude Sonnet 5": "anthropic.claude-sonnet-5-v1",
    "Claude Opus 4.8": "anthropic.claude-opus-4-8-v1",
    "Claude Opus 4.7": "anthropic.claude-opus-4-7-v1",
    "Claude Opus 4.6": "anthropic.claude-opus-4-6-v1",
    "Claude Sonnet 4.6": "anthropic.claude-sonnet-4-6",
    "Claude Opus 4.5": "anthropic.claude-opus-4-5-v1",
    "Claude Sonnet 4.5": "anthropic.claude-sonnet-4-5-v1",
    "Claude Haiku 4.5": "anthropic.claude-haiku-4-5-v1",
    "Claude Sonnet 4": "anthropic.claude-sonnet-4-v1",
    "Claude Opus 4": "anthropic.claude-opus-4-20250514-v1",
    "Claude Opus 4.1": "anthropic.claude-opus-4-1-v1",
    "Claude 3.7 Sonnet": "anthropic.claude-3-7-sonnet-20250219-v1",
    "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-20241022-v2",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1",
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
    "Nova MME": "amazon.nova-2-multimodal-embeddings-v1",
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
    "Pixtral Large 25.02": "mistral.pixtral-large-2502",
    # Nvidia
    "NVIDIA Nemotron Nano 2 VL": "nvidia.nemotron-nano-12b-v2-vl",
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

# Models with a known future list-price change. Maps a model-id substring to the
# date the new rate takes effect and its multiplier vs the current (base) rate.
# The base tier stays the AWS-reported rate; the scheduled override is derived
# from it. Populate this only from an announcement AWS has actually published.
DATED_PRICE_SCHEDULES: dict[str, tuple[date, Decimal]] = {}

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

# Non-mantle display names for models the Foundation Models API prices under a different
# servicename: AWS bills Claude 2.0 and 2.1 as the single "Claude" -> anthropic.claude-v2.
# Deliberately not NON_MANTLE_MODEL_MAP entries, which would emit a second conflicting key.
NON_MANTLE_PRICED_AS_FM = frozenset({"Claude 2.0", "Claude 2.1"})

# Embedding models (output_cost_per_token = 0.0)
EMBEDDING_PREFIXES = (
    "amazon.titan-embed",
    "amazon.nova-embed",
    "amazon.nova-2-multimodal-embed",
    "cohere.embed",
)

# Cross-region inference profile prefixes (excluding "global." which gets its own pricing)
INFERENCE_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "au.", "jp.", "ca.")

# ── Bedrock API integration ──────────────────────────────────────────────────

_DATE_IN_ID = re.compile(r"-\d{8}")
_COLON_VERSION = re.compile(r":\d+$")
_DASH_VERSION = re.compile(r"-v\d+$")
_INSTRUCT_SUFFIX = re.compile(r"-instruct$")
_THROUGHPUT_VARIANT = re.compile(r":\d+:\w")


def _strip_colon_version(model_id: str) -> str:
    """Strip :N version suffix (e.g. :0) from a model ID."""
    return _COLON_VERSION.sub("", model_id)


def _strip_date_from_id(model_id: str) -> str:
    """Strip date component (e.g. -20251101) from a model ID."""
    return _DATE_IN_ID.sub("", model_id)


# Representative regions for discovering geo-specific inference profiles.
# One region per geo prefix — queried to find all profiles for that prefix.
GEO_DISCOVERY_REGIONS: dict[str, str] = {
    "us-east-1": "us.",
    "eu-central-1": "eu.",
    "ap-southeast-1": "apac.",
    "ap-southeast-2": "au.",
    "ap-northeast-1": "jp.",
    "ca-central-1": "ca.",
}


def fetch_bedrock_catalog() -> tuple[list[str], list[str]]:
    """Fetch foundation model and inference profile IDs from the Bedrock API.

    Queries us-east-1 for foundation models and global profiles, then queries
    one representative region per geo to discover all regional inference profiles.

    Uses boto3's default credential chain (env vars, AWS config, instance metadata).
    """
    session = boto3.Session()

    # Foundation models + global/US profiles from us-east-1
    client = session.client("bedrock", region_name="us-east-1")
    models_resp = client.list_foundation_models()
    model_ids = [m["modelId"] for m in models_resp["modelSummaries"]]

    all_profile_ids: set[str] = set()
    for region, geo_prefix in GEO_DISCOVERY_REGIONS.items():
        try:
            client = session.client("bedrock", region_name=region)
            resp = client.list_inference_profiles()
            region_profiles = [p["inferenceProfileId"] for p in resp["inferenceProfileSummaries"]]
            # Only keep profiles matching this geo's prefix (+ global from us-east-1)
            for pid in region_profiles:
                stripped = _strip_colon_version(pid)
                if stripped.startswith((geo_prefix, "global.")):
                    all_profile_ids.add(pid)
            geo_count = sum(1 for p in region_profiles if _strip_colon_version(p).startswith(geo_prefix))
            _info(f"  {region}: {geo_count} {geo_prefix}* profiles")
        except (ClientError, BotoCoreError, EndpointConnectionError) as e:
            _warn(f"Failed to query {region} for {geo_prefix}* profiles: {e}")

    return model_ids, sorted(all_profile_ids)


def _normalize_model_id(model_id: str) -> str:
    """Strip date, -vN, and -instruct suffixes from a model ID."""
    return _INSTRUCT_SUFFIX.sub("", _DASH_VERSION.sub("", _strip_date_from_id(model_id)))


def _build_resolution_indexes(
    real_model_ids: list[str],
) -> tuple[set[str], set[str], dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    """Build lookup indexes from real Bedrock model IDs for resolution matching.

    Returns (real_raw, real_clean, dateless_to_real, normalized_to_real, prefix_to_real).
    """
    real_raw: set[str] = set()
    real_clean: set[str] = set()
    for rid in real_model_ids:
        if _THROUGHPUT_VARIANT.search(rid):
            continue
        real_raw.add(rid)
        real_clean.add(_strip_colon_version(rid))

    dateless_to_real: dict[str, list[str]] = {}
    normalized_to_real: dict[str, list[str]] = {}
    prefix_to_real: dict[str, str] = {}

    for clean in real_clean:
        dateless = _strip_date_from_id(clean)
        if dateless != clean:
            dateless_to_real.setdefault(dateless, []).append(clean)

        normalized = _normalize_model_id(clean)
        if normalized != clean:
            normalized_to_real.setdefault(normalized, []).append(clean)

        # Reverse prefix: part before last dash segment (e.g. gpt-oss-120b for gpt-oss-120b-1)
        base = clean.rsplit("-", 1)[0]
        if base and base not in real_clean:
            _ = prefix_to_real.setdefault(base, clean)

    return real_raw, real_clean, dateless_to_real, normalized_to_real, prefix_to_real


# Simplified IDs that never resolve, because AWS no longer lists the model. Membership records
# that someone checked what a caller actually presents -- not that the key is necessarily right.
# An unresolved ID missing from this set is a newly retired model whose key may have silently
# become unreachable, so generation stops instead of emitting an entry nothing can look up.
KNOWN_UNRESOLVED_IDS: frozenset[str] = frozenset(
    {
        # The key is itself the real (retired) Bedrock model ID, so its entry is reachable.
        "ai21.j2-mid-v1",
        "ai21.j2-ultra-v1",
        "ai21.jamba-instruct-v1",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-text-premier-v1",
        "anthropic.claude-instant-v1",
        "anthropic.claude-v2",
        "cohere.command-light-text-v14",
        "cohere.command-r-plus-v1",
        "cohere.command-r-v1",
        "cohere.command-text-v14",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "meta.llama3-2-11b-instruct-v1",
        "meta.llama3-2-1b-instruct-v1",
        "meta.llama3-2-3b-instruct-v1",
        "meta.llama3-2-90b-instruct-v1",
        # Reachable: the key is a prefix of the real ID, but the model is not offered in
        # us-east-1 and fetch_bedrock_catalog only lists that Region, so it cannot be confirmed.
        "qwen.qwen3-235b-a22b-2507",
        # Dated IDs hardcoded in FM_SERVICENAME_MAP; delisted, so they cannot resolve.
        "anthropic.claude-3-5-haiku-20241022-v1",
        "anthropic.claude-3-5-sonnet-20240620-v1",
        "anthropic.claude-3-5-sonnet-20241022-v2",
        "anthropic.claude-3-7-sonnet-20250219-v1",
        "anthropic.claude-3-opus-20240229-v1",
        "anthropic.claude-3-sonnet-20240229-v1",
        "anthropic.claude-opus-4-20250514-v1",
        # Not confirmed reachable: the real ID may differ, which would price these as None.
        # These are live models rather than retirements, tracked as a separate known gap.
        "amazon.nova-2-omni-v1",
        "amazon.nova-2-pro-v1",
        "anthropic.claude-mythos-5-1",
        "anthropic.claude-mythos-5-v1",
        "deepseek.v3.1",
        "google.gemma-4-26b-a4b",
        "google.gemma-4-31b",
        "google.gemma-4-e2b",
        "moonshotai.kimi-k2-thinking",
        "nvidia.nemotron-nano-12b-v2-vl",
        "qwen.qwen3-coder-480b-a35b-instruct",
        "xai.grok-4.3",
        "zai.glm5",
    }
)


def build_id_resolution_map(
    simplified_ids: set[str],
    real_model_ids: list[str],
) -> dict[str, str]:
    """Map simplified pricing IDs to real Bedrock model IDs.

    Tries four strategies:
    1. Strip date from real ID, match against simplified (handles dated models)
    2. Strip -vN from simplified, match against real (handles version-less models)
    3. Normalize both (strip date, -vN, -instruct), match (handles suffix mismatches)
    4. Prefix match: simplified is a prefix of a real ID (handles extra segments)
    """
    real_raw, real_clean, dateless_to_real, normalized_to_real, prefix_to_real = _build_resolution_indexes(
        real_model_ids
    )

    mapping: dict[str, str] = {}
    unresolved: list[str] = []

    for sid in simplified_ids:
        if sid in real_clean or sid in real_raw:
            continue

        # Strategy 1: dateless form of a real ID matches simplified
        candidates = dateless_to_real.get(sid)
        if candidates:
            mapping[sid] = sorted(candidates)[-1]
            continue

        # Strategy 2: simplified minus -vN matches a real ID
        sid_no_v = _DASH_VERSION.sub("", sid)
        if sid_no_v != sid and sid_no_v in real_clean:
            mapping[sid] = sid_no_v
            continue

        # Strategy 3: fully normalized match
        sid_normalized = _normalize_model_id(sid)
        if sid_normalized != sid and sid_normalized in real_clean:
            mapping[sid] = sid_normalized
            continue
        candidates = normalized_to_real.get(sid_normalized)
        if candidates:
            mapping[sid] = sorted(candidates)[-1]
            continue

        # Strategy 4: simplified is a prefix of a real ID
        if sid in prefix_to_real:
            mapping[sid] = prefix_to_real[sid]
            continue

        unresolved.append(sid)

    unexpected = sorted(set(unresolved) - KNOWN_UNRESOLVED_IDS)
    if unexpected:
        _die(
            f"Could not resolve to real Bedrock model IDs: {unexpected}. A model that retires drops "
            "out of list_foundation_models, so its simplified key stops resolving and would be "
            "written out as an entry no caller can look up. Point FM_SERVICENAME_MAP at the real "
            "dated model ID, then record the key in KNOWN_UNRESOLVED_IDS."
        )
    if unresolved:
        _info(f"Unresolved but known ({len(unresolved)}): {sorted(unresolved)}")
    if mapping:
        _info(f"Resolved {len(mapping)} simplified IDs to real Bedrock model IDs:")
        for old, new in sorted(mapping.items()):
            _info(f"  {old} -> {new}")

    return mapping


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ModelPrices:
    """Intermediate representation of a model's pricing (per million tokens)."""

    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    cache_read_cost: Decimal | None = None
    cache_write_cost: Decimal | None = None
    cache_write_1h_cost: Decimal | None = None
    # Long-context tier (>200K tokens)
    lctx_input_cost: Decimal | None = None
    lctx_output_cost: Decimal | None = None
    lctx_cache_read_cost: Decimal | None = None
    lctx_cache_write_cost: Decimal | None = None
    lctx_cache_write_1h_cost: Decimal | None = None

    @property
    def has_lctx(self) -> bool:
        return self.lctx_input_cost is not None


def resolve_pricing_ids(
    pricing: dict[str, ModelPrices],
    resolution_map: dict[str, str],
) -> dict[str, ModelPrices]:
    """Re-key pricing dict from simplified IDs to real Bedrock model IDs."""
    return {resolution_map.get(k, k): v for k, v in pricing.items()}


def expand_with_real_profiles(
    default: dict[str, ModelPrices],
    global_pricing: dict[str, ModelPrices],
    real_profile_ids: list[str],
) -> dict[str, ModelPrices]:
    """Expand pricing with real inference profile IDs from Bedrock.

    For global.* profiles, uses global pricing. For regional profiles (us.*, eu.*, etc.),
    uses default (non-global) pricing.
    """
    result = dict(default)

    for pid in real_profile_ids:
        clean_pid = _strip_colon_version(pid)

        # Determine prefix and base model
        is_global = False
        base: str | None = None
        for pfx in ("global.", *INFERENCE_PROFILE_PREFIXES):
            if clean_pid.startswith(pfx):
                base = clean_pid[len(pfx) :]
                is_global = pfx == "global."
                break

        if base is None:
            continue  # Not a recognized inference profile prefix

        # Find pricing for base model
        if is_global and base in global_pricing:
            result[clean_pid] = global_pricing[base]
        elif base in default:
            result[clean_pid] = default[base]

    return result


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
    """Parse mantle entries from AmazonBedrock API.

    Token prices are scaled to per-million based on each dimension's ``unit``
    (usually ``1K tokens``, but AWS ships ``1M tokens`` for some models such as
    xAI Grok).
    """
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

        priced = _get_price_with_unit(sku, terms)
        if priced is None:
            continue
        price, unit = priced

        price_per_m = _scale_to_per_million(price, unit)
        if price_per_m is None:
            _warn(f"Unrecognized price unit {unit!r} for {ut}; skipping dimension")
            continue

        if model_id not in result:
            result[model_id] = ModelPrices()

        mp = result[model_id]
        if dimension_part == "input-tokens":
            mp.input_cost = price_per_m
        elif dimension_part == "output-tokens":
            mp.output_cost = price_per_m
        elif dimension_part in ("cache-read-input-tokens", "cache-read-tokens"):
            mp.cache_read_cost = price_per_m
        elif dimension_part in ("cache-write-input-tokens", "cache-write-tokens"):
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
    elif dim_name == "cache_write_1h":
        mp.cache_write_1h_cost = price


def _collect_non_mantle_prices(
    products: dict[str, Any], terms: dict[str, Any]
) -> tuple[dict[str, dict[str, dict[bool, Decimal]]], set[str]]:
    """Collect non-mantle prices as model_id -> dimension -> {is_global: price}.

    Also returns the display names dropped for want of a ``NON_MANTLE_MODEL_MAP`` entry.
    """
    # Most models priced elsewhere also publish non-mantle meters here: mantle models carry
    # the same `model` attribute, and Foundation Models are keyed by the same display name.
    # Neither is a NON_MANTLE_MODEL_MAP gap, so neither is reported as unmapped.
    priced_elsewhere = (
        {
            (prod.get("attributes", {}).get("model") or "").strip()
            for prod in products.values()
            if "-mantle-" in prod.get("attributes", {}).get("usagetype", "")
        }
        | set(FM_SERVICENAME_MAP)
        | NON_MANTLE_PRICED_AS_FM
    )

    collected: dict[str, dict[str, dict[bool, Decimal]]] = {}
    unmapped: set[str] = set()

    for sku, prod in products.items():
        attrs = prod.get("attributes", {})
        ut = attrs.get("usagetype", "")

        if "-mantle-" in ut or _should_skip_usagetype(ut):
            continue

        dimension = _classify_dimension(attrs.get("inferenceType", ""), ut)
        if dimension is None:
            continue

        model_name = attrs.get("model", "").strip()
        model_id = _resolve_non_mantle_model_id(model_name, ut)
        if model_id is None:
            if model_name and model_name not in priced_elsewhere:
                unmapped.add(model_name)
            continue

        priced = _get_price_with_unit(sku, terms)
        if priced is None:
            continue
        price, unit = priced

        price_per_m = _scale_to_per_million(price, unit)
        if price_per_m is None:
            _warn(f"Unrecognized price unit {unit!r} for {ut}; skipping dimension")
            continue
        collected.setdefault(model_id, {}).setdefault(dimension, {})[_is_global_usagetype(ut)] = price_per_m

    return collected, unmapped


def parse_amazon_models(
    data: dict[str, Any], *, fallback_to_global: bool = True, report_unmapped: bool = False
) -> tuple[dict[str, ModelPrices], dict[str, ModelPrices]]:
    """Parse non-mantle entries from AmazonBedrock API for Amazon + legacy models.

    Returns (default_pricing, global_pricing) where global_pricing contains only
    models that have cross-region global inference profile pricing.

    ``fallback_to_global`` fills a missing standard rate from the global rate. Correct for the
    us-east-1 baseline (a global-only model still needs a list price), but wrong for regional diffs:
    a Region that publishes only the global meter has no genuine standard rate, and emitting the
    global discount as its standard rate under-reports geo-profile calls (see ``_fetch_regional_diffs``).

    ``report_unmapped`` warns about models this parser dropped for want of a
    ``NON_MANTLE_MODEL_MAP`` entry, mirroring the unmapped-servicename warning in
    :func:`parse_foundation_models`. Enabled for the us-east-1 baseline only: a Region that
    publishes a meter the baseline already priced elsewhere is not a missing model.
    """
    collected, unmapped = _collect_non_mantle_prices(
        data.get("products", {}), data.get("terms", {}).get("OnDemand", {})
    )

    if report_unmapped and unmapped:
        _warn(f"Unmapped non-mantle model names: {sorted(unmapped)}")

    # Build result: separate default (non-global) and global pricing
    result: dict[str, ModelPrices] = {}
    global_result: dict[str, ModelPrices] = {}
    for model_id, dims in collected.items():
        default_mp = ModelPrices()
        global_mp = ModelPrices()
        has_global = False
        for dim_name, prices_by_scope in dims.items():
            non_global_price = prices_by_scope.get(False)
            global_price = prices_by_scope.get(True)
            price = non_global_price if non_global_price is not None else (global_price if fallback_to_global else None)
            if price is not None:
                _set_dimension(default_mp, dim_name, price)
            if global_price is not None:
                _set_dimension(global_mp, dim_name, global_price)
                has_global = True
        if default_mp.input_cost is not None:
            result[model_id] = default_mp
        if has_global and global_mp.input_cost is not None:
            global_result[model_id] = global_mp

    return result, global_result


# ── Parsing: AmazonBedrockFoundationModels ───────────────────────────────────


def _parse_fm_dimension(usagetype: str) -> tuple[str, bool] | None:
    """Extract (dimension, is_lctx) from a Foundation Models usagetype.

    Supports two naming schemes:
    - Legacy PascalCase: ``{REGION}-MP:{REGION}_{Dimension}[_{Variant}]-Units``
      e.g. ``USE1-MP:USE1_InputTokenCount_Global-Units``
    - New snake_case (Claude Opus 4.7+): ``{REGION}-MP:{REGION}_{token_kind}[_1h][_global]_standard-Units``
      e.g. ``USE1-MP:USE1_cache_write_tokens_global_standard-Units``

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
        # Both spellings are required: legacy PascalCase is "_Batch", the snake_case
        # form used from Claude Opus 5 onward is "_batch". Missing the lowercase one
        # lets a batch rate overwrite the standard rate under the same dimension key.
        "Batch",
        "_batch",
        "LatencyOptimized",
        "ModelStorage",
        "Customization",
        "search_units",
        "MillionBatch",
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

    is_lctx = "_LCtx" in field or "_lctx" in field

    # 1h cache-write dimensions need handling BEFORE the pattern table: the legacy
    # field is "CacheWrite1hInputTokenCount", which does NOT match the
    # CacheWriteInputTokenCount pattern but WOULD partial-match InputTokenCount
    # and misclassify a 2x cache write as plain input.
    if "CacheWrite1h" in field or "_1h" in field:
        if "CacheWrite1h" in field or "cache_write_tokens" in field:
            return ("cache_write_1h", is_lctx)
        return None  # unknown 1h dimension — skip rather than misclassify

    # Determine dimension (order matters: CacheRead/CacheWrite before Input/Output
    # so PascalCase CacheReadInputTokenCount isn't partial-matched by InputTokenCount).
    dimension_patterns = [
        # Legacy PascalCase
        ("CacheReadInputTokenCount", "cache_read"),
        ("CacheWriteInputTokenCount", "cache_write"),
        ("InputTokenCount", "input"),
        ("OutputTokenCount", "output"),
        # New snake_case
        ("cache_read_tokens", "cache_read"),
        ("cache_write_tokens", "cache_write"),
        ("input_tokens", "input"),
        ("output_tokens", "output"),
    ]
    return next(
        ((dim, is_lctx) for pattern, dim in dimension_patterns if pattern in field),
        None,
    )


def _is_global_fm(usagetype: str) -> bool:
    """Whether this FM usagetype is Global (cross-region) pricing.

    Legacy PascalCase uses ``_Global`` (but ``_Global_Batch`` is batch, not global-standard).
    New snake_case (Opus 4.7+) uses lowercase ``_global_`` before ``_standard``.
    """
    dim_part = usagetype.split("-MP:")[1] if "-MP:" in usagetype else ""
    legacy_global = "_Global" in dim_part and "_Batch" not in dim_part
    snake_global = "_global_" in dim_part
    return legacy_global or snake_global


def parse_foundation_models(
    data: dict[str, Any], *, fallback_to_global: bool = True
) -> tuple[dict[str, ModelPrices], dict[str, ModelPrices]]:
    """Parse AmazonBedrockFoundationModels API. Prices are per million tokens.

    Returns (default_pricing, global_pricing) where global_pricing contains only
    models that have cross-region global inference profile pricing.

    ``fallback_to_global`` fills a missing standard rate from the global rate -- see
    :func:`parse_amazon_models`; it is disabled for regional diffs.
    """
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

    return _build_fm_result(collected, fallback_to_global=fallback_to_global)


def _build_fm_result(
    collected: dict[str, dict[tuple[str, bool, bool], Decimal]],
    *,
    fallback_to_global: bool = True,
) -> tuple[dict[str, ModelPrices], dict[str, ModelPrices]]:
    """Build ModelPrices from collected Foundation Models data.

    Returns (default_pricing, global_pricing). ``default`` uses non-global (standard) prices;
    ``fallback_to_global`` fills a missing standard dimension from the global rate (see
    :func:`parse_amazon_models`). ``global_pricing`` contains only models with global pricing.
    """
    result: dict[str, ModelPrices] = {}
    global_result: dict[str, ModelPrices] = {}
    for model_id, prices in collected.items():
        default_mp = ModelPrices()
        global_mp = ModelPrices()
        has_global = False
        for dim_name in ("input", "output", "cache_read", "cache_write", "cache_write_1h"):
            non_global_std = prices.get((dim_name, False, False))
            global_std = prices.get((dim_name, False, True))
            std = non_global_std if non_global_std is not None else (global_std if fallback_to_global else None)
            if std is not None:
                _set_dimension(default_mp, dim_name, std)
            if global_std is not None:
                _set_dimension(global_mp, dim_name, global_std)
                has_global = True

            # Long-context tier: same pattern
            non_global_lctx = prices.get((dim_name, True, False))
            global_lctx = prices.get((dim_name, True, True))
            lctx = non_global_lctx if non_global_lctx is not None else (global_lctx if fallback_to_global else None)
            _set_fm_lctx(default_mp, dim_name, lctx)
            if global_lctx is not None:
                _set_fm_lctx(global_mp, dim_name, global_lctx)

        if default_mp.input_cost is not None:
            result[model_id] = default_mp
        if has_global and global_mp.input_cost is not None:
            global_result[model_id] = global_mp
    return result, global_result


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
    elif dim_name == "cache_write_1h":
        mp.lctx_cache_write_1h_cost = price


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


def _tier_is_emittable(input_cost: Decimal | None, output_cost: Decimal | None, *, is_embedding: bool) -> bool:
    """Whether a tier has the rates ``PricingTier`` requires. Embeddings bill no output tokens."""
    if input_cost is None:
        return False
    return is_embedding or output_cost is not None


def drop_unemittable(prices: dict[str, ModelPrices]) -> tuple[dict[str, ModelPrices], list[tuple[str, str]]]:
    """Split overrides into those safe to emit and the (model ID, reason) pairs that are not.

    Input and output rates are mandatory -- a ``PricingTier`` will not construct without them -- so a
    Region that omits either (e.g. ap-east-2 lists ``APE2-NovaLite-input-tokens`` with no non-batch
    ``-output-tokens``) is dropped and falls back to us-east-1. A Region that prices input/output but
    omits a cache dimension (e.g. eu-west-2 Nova Pro, whose cache reads are priced only on the flex
    and priority tiers) is kept: its genuine input/output premium is emitted, and a cached call
    against the missing rate is reported as unpriced at runtime -- see
    ``lmux.cost.usage_has_unpriced_dimension`` -- rather than billed for free.
    """
    keep: dict[str, ModelPrices] = {}
    dropped: list[tuple[str, str]] = []
    for model_id, mp in prices.items():
        is_emb = _is_embedding(model_id)
        if not _tier_is_emittable(mp.input_cost, mp.output_cost, is_embedding=is_emb):
            dropped.append((model_id, "no complete on-demand tier"))
            continue
        if mp.has_lctx and not _tier_is_emittable(mp.lctx_input_cost, mp.lctx_output_cost, is_embedding=is_emb):
            dropped.append((model_id, "no complete long-context tier"))
            continue
        keep[model_id] = mp
    return keep, dropped


def _prices_differ(a: ModelPrices, b: ModelPrices) -> bool:
    """Whether two ModelPrices differ in any field (standard or long-context tier)."""
    return (
        a.input_cost != b.input_cost
        or a.output_cost != b.output_cost
        or a.cache_read_cost != b.cache_read_cost
        or a.cache_write_cost != b.cache_write_cost
        or a.cache_write_1h_cost != b.cache_write_1h_cost
        or a.lctx_input_cost != b.lctx_input_cost
        or a.lctx_output_cost != b.lctx_output_cost
        or a.lctx_cache_read_cost != b.lctx_cache_read_cost
        or a.lctx_cache_write_cost != b.lctx_cache_write_cost
        or a.lctx_cache_write_1h_cost != b.lctx_cache_write_1h_cost
    )


# ── Code generation ──────────────────────────────────────────────────────────


def _strip_profile_prefix(model_id: str) -> str:
    """Strip inference profile prefix (us., eu., global., etc.) from a model ID."""
    for pfx in ("global.", *INFERENCE_PROFILE_PREFIXES):
        if model_id.startswith(pfx):
            return model_id[len(pfx) :]
    return model_id


def _get_provider_group(model_id: str) -> str:
    """Get the provider group name for a model ID."""
    bare_id = _strip_profile_prefix(model_id)
    for prefix, group_name in PROVIDER_GROUPS:
        if bare_id.startswith(prefix):
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
    bare_id = _strip_profile_prefix(model_id)
    return any(bare_id.startswith(p) for p in EMBEDDING_PREFIXES)


def generate_cost_py(
    pricing: dict[str, ModelPrices],
    regional: dict[str, dict[str, ModelPrices]] | None = None,
) -> str:
    """Generate the complete cost.py source code."""
    lines: list[str] = []
    has_dated = any(_dated_schedule_for(mid) for mid in pricing) or any(
        _dated_schedule_for(mid) for region in (regional or {}).values() for mid in region
    )
    _emit_header(lines, has_regional=bool(regional), has_dated=has_dated)
    _emit_pricing_dict(lines, pricing)
    _emit_regional_dict(lines, regional)
    _emit_function(lines)
    return "\n".join(lines)


def _emit_import_lines(lines: list[str], *, has_dated: bool) -> None:
    """Emit the datetime + lmux imports common to both generated pricing modules."""
    lines.append("from datetime import date")
    lines.append("")
    names = ["ModelPricing", "PricingTier", "calculate_cost", "per_million_tokens"]
    if has_dated:
        names.insert(1, "PricingSchedule")
    lines.append("from lmux.cost import " + ", ".join(names))
    lines.append("from lmux.types import Cost, Usage")


def _emit_header(lines: list[str], *, has_regional: bool, has_dated: bool) -> None:
    """Emit the module docstring and imports for the non-Anthropic Bedrock cost.py.

    Anthropic-on-Bedrock pricing is emitted separately into lmux_bedrock_shared.pricing
    (see :func:`generate_shared_anthropic_py`) and merged back into ``_PRICING`` below.
    """
    lines.append('"""AWS Bedrock pricing data and cost calculation.')
    lines.append("")
    lines.append("Prices are for the us-east-1 region (on-demand, cross-region global inference).")
    if has_regional:
        lines.append("Regional pricing overrides are included for regions where prices differ.")
    else:
        lines.append("Use register_pricing() on BedrockProvider for overrides or other regions.")
    lines.append("")
    lines.append("Anthropic Claude pricing lives in lmux_bedrock_shared.pricing (shared with the")
    lines.append("native lmux-anthropic Bedrock provider) and is merged into _PRICING below.")
    lines.append("")
    lines.append("Auto-generated by scripts/update_bedrock_pricing.py -- do not edit manually.")
    lines.append("")
    lines.append("Pricing source: https://aws.amazon.com/bedrock/pricing/")
    lines.append('"""')
    lines.append("")
    _emit_import_lines(lines, has_dated=has_dated)
    lines.append("from lmux_bedrock_shared.pricing import (")
    lines.append("    ANTHROPIC_PRICING,")
    lines.append("    ANTHROPIC_REGIONAL_PRICING,")
    lines.append("    DEFAULT_PRICING_REGION,")
    lines.append("    INFERENCE_PROFILE_PREFIXES,")
    lines.append("    cost_or_none,")
    lines.append("    lookup_pricing,")
    lines.append("    lookup_regional_pricing,")
    lines.append(")")
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

    lines.append("    **ANTHROPIC_PRICING,")
    lines.append("}")
    lines.append("")


def _emit_nested_pricing_dict(
    lines: list[str],
    name: str,
    regional: dict[str, dict[str, ModelPrices]] | None,
    *,
    comment: str,
) -> None:
    """Emit a ``{region: {model_id: ModelPricing}}`` literal under ``name``."""
    if regional and any(regional.values()):
        lines.append(f"# {comment}")
        lines.append(f"{name}: dict[str, dict[str, ModelPricing]] = {{")
        for region in sorted(regional.keys()):
            if not regional[region]:
                continue
            lines.append(f'    "{region}": {{')
            for model_id in sorted(regional[region].keys()):
                _emit_model_pricing(lines, model_id, regional[region][model_id], indent=8)
            lines.append("    },")
        lines.append("}")
    else:
        lines.append(f"{name}: dict[str, dict[str, ModelPricing]] = {{}}")
    lines.append("")


def _emit_regional_dict(
    lines: list[str],
    regional: dict[str, dict[str, ModelPrices]] | None,
) -> None:
    """Emit the non-Anthropic regional overrides merged with the shared Anthropic subset."""
    _emit_nested_pricing_dict(
        lines,
        "_BEDROCK_REGIONAL",
        regional,
        comment="Regional pricing overrides (only models that differ from us-east-1)",
    )
    lines.append("# Claude's overrides come from the shared table, so both Bedrock providers price it identically.")
    lines.append("_REGIONAL_PRICING: dict[str, dict[str, ModelPricing]] = {")
    lines.append("    region: {**_BEDROCK_REGIONAL.get(region, {}), **ANTHROPIC_REGIONAL_PRICING.get(region, {})}")
    lines.append("    for region in _BEDROCK_REGIONAL.keys() | ANTHROPIC_REGIONAL_PRICING.keys()")
    lines.append("}")
    lines.append("")
    lines.append("# Pre-sorted by key length descending for longest-prefix matching")
    lines.append("_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)")
    lines.append("")
    lines.append("")


_FUNCTION_BODY = """\
def calculate_bedrock_cost(
    model: str, usage: Usage, *, region: str | None = None, as_of: date | None = None
) -> Cost | None:
    \"\"\"Calculate cost for a Bedrock API call. Returns None if model pricing is unknown.

    ``region`` is the Region the request is sent to, which is the Region Bedrock bills against --
    a cross-Region inference profile is priced by the Region it is called from, not by wherever the
    request is routed. ``as_of`` selects dated pricing for models with scheduled rate changes;
    it defaults to the latest schedule. See ``lmux.cost.calculate_cost``.

    Matching is shared with the native Anthropic Bedrock provider (see
    ``lmux_bedrock_shared.pricing``) so Claude resolves identically through either.
    \"\"\"
    if region is not None and region != DEFAULT_PRICING_REGION:
        pricing = lookup_regional_pricing(_REGIONAL_PRICING, region, model)
        if pricing is not None:
            return cost_or_none(pricing, usage, as_of)

    pricing = lookup_pricing(_PRICING, _PRICING_BY_PREFIX, model, INFERENCE_PROFILE_PREFIXES)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing, as_of)
"""


def _emit_function(lines: list[str]) -> None:
    """Emit the calculate_bedrock_cost function."""
    lines.extend(_FUNCTION_BODY.splitlines())
    lines.append("")


def _split_regional(
    regional: dict[str, dict[str, ModelPrices]] | None,
    *,
    anthropic: bool,
) -> dict[str, dict[str, ModelPrices]]:
    """Take either the Anthropic or the non-Anthropic half of the regional overrides."""
    split: dict[str, dict[str, ModelPrices]] = {}
    for region, models in (regional or {}).items():
        half = {mid: mp for mid, mp in models.items() if _is_anthropic(mid) is anthropic}
        if half:
            split[region] = half
    return split


def _is_anthropic(model_id: str) -> bool:
    """Whether a (possibly profile-prefixed) model ID is an Anthropic Claude model."""
    return _strip_profile_prefix(model_id).startswith("anthropic.")


_SHARED_ANTHROPIC_FUNCTION_BODY = '''\
_ANTHROPIC_BY_PREFIX = sorted(ANTHROPIC_PRICING.items(), key=lambda item: len(item[0]), reverse=True)

# The Region the default tables are priced for; every other Region is an override.
DEFAULT_PRICING_REGION = "us-east-1"

# Cross-region inference profile prefixes. Bedrock bills a profile call at the standard rate of the
# Region the request is sent to, so a geo profile (e.g. "us.anthropic...") with no dedicated entry
# resolves to the base model. "global." is priced separately (~10% below standard) and is never
# resolved to the base model inside a regional table: absent there means it matches the default
# table, not that it takes the Region's standard rate.
GEO_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "au.", "jp.", "ca.")
GLOBAL_PROFILE_PREFIX = "global."
INFERENCE_PROFILE_PREFIXES = (GLOBAL_PROFILE_PREFIX, *GEO_PROFILE_PREFIXES)


def strip_profile_prefix(model: str, prefixes: tuple[str, ...] = INFERENCE_PROFILE_PREFIXES) -> str:
    """Drop an inference-profile prefix from a model ID, if it carries one."""
    for prefix in prefixes:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def lookup_pricing(
    table: dict[str, ModelPricing],
    by_prefix: list[tuple[str, ModelPricing]],
    model: str,
    strip_prefixes: tuple[str, ...],
) -> ModelPricing | None:
    """Exact match, then longest-prefix, then one retry against the profile-stripped ID."""
    pricing = table.get(model)
    if pricing is not None:
        return pricing
    for prefix, p in by_prefix:
        if model.startswith(prefix):
            return p
    bare = strip_profile_prefix(model, strip_prefixes)
    if bare != model:
        return lookup_pricing(table, by_prefix, bare, strip_prefixes)
    return None


def lookup_regional_pricing(
    regional: dict[str, dict[str, ModelPricing]], region: str, model: str
) -> ModelPricing | None:
    """A Region's pricing override, or None when it has none and the default table applies.

    Sorted per call rather than precomputed: the prefix order has to come from the same table the
    exact match reads, or the two can disagree. Only one Region's handful of overrides is sorted,
    on a request that already carries an HTTP round trip.
    """
    table = regional.get(region)
    if not table:
        return None
    by_prefix = sorted(table.items(), key=lambda item: len(item[0]), reverse=True)
    return lookup_pricing(table, by_prefix, model, GEO_PROFILE_PREFIXES)


def calculate_bedrock_anthropic_cost(
    model: str, usage: Usage, *, region: str | None = None, as_of: date | None = None
) -> Cost | None:
    """Calculate cost for a native Anthropic-on-Bedrock call. None if pricing is unknown.

    Handles both bare model IDs and cross-region inference-profile IDs
    (e.g. ``us.anthropic.claude-opus-4-8``). ``region`` is the Region the request is sent to, which
    is the Region Bedrock bills against -- a cross-Region inference profile is priced by the Region
    it is called from, not by wherever the request is routed. ``as_of`` selects dated pricing for
    models with scheduled rate changes; it defaults to the latest schedule.
    See ``lmux.cost.calculate_cost``.
    """
    if region is not None and region != DEFAULT_PRICING_REGION:
        pricing = lookup_regional_pricing(ANTHROPIC_REGIONAL_PRICING, region, model)
        if pricing is not None:
            return cost_or_none(pricing, usage, as_of)
    pricing = lookup_pricing(ANTHROPIC_PRICING, _ANTHROPIC_BY_PREFIX, model, INFERENCE_PROFILE_PREFIXES)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing, as_of)


def cost_or_none(pricing: ModelPricing, usage: Usage, as_of: date | None = None) -> Cost | None:
    """Cost from a regional override, or None when it leaves a billed token dimension unpriced.

    A Region may publish input/output but no cache meter; ``calculate_cost`` treats a missing rate
    as zero, which would bill those cache tokens for free, so the unknown cost is reported as None
    instead. Regional overrides are single-tier, so the base tier's rates decide.
    """
    tier = pricing.tiers[0]
    creation = max(usage.cache_creation_tokens or 0, sum((usage.cache_creation_tokens_by_ttl or {}).values()))
    if (usage.cache_read_tokens or 0) and tier.cache_read_cost_per_token is None:
        return None
    if creation and tier.cache_creation_cost_per_token is None and not tier.cache_creation_cost_per_token_by_ttl:
        return None
    return calculate_cost(usage, pricing, as_of)
'''


def _emit_shared_header(lines: list[str], *, has_dated: bool) -> None:
    """Emit the module docstring and imports for lmux_bedrock_shared.pricing."""
    lines.append('"""AWS Bedrock pricing for Anthropic Claude models.')
    lines.append("")
    lines.append("Anthropic-on-Bedrock pricing, shared by lmux-aws-bedrock (Converse) and the")
    lines.append("native lmux-anthropic Bedrock provider so Claude is priced identically by both.")
    lines.append("Only the Anthropic subset lives here; other Bedrock vendors are priced in")
    lines.append("lmux_aws_bedrock.cost.")
    lines.append("")
    lines.append("Auto-generated by scripts/update_bedrock_pricing.py -- do not edit manually.")
    lines.append("")
    lines.append("Pricing source: https://aws.amazon.com/bedrock/pricing/")
    lines.append('"""')
    lines.append("")
    _emit_import_lines(lines, has_dated=has_dated)
    lines.append("")


def generate_shared_anthropic_py(
    pricing: dict[str, ModelPrices],
    regional: dict[str, dict[str, ModelPrices]] | None = None,
) -> str:
    """Generate the lmux_bedrock_shared.pricing source (Anthropic-on-Bedrock subset)."""
    lines: list[str] = []
    has_dated = any(_dated_schedule_for(mid) for mid in pricing) or any(
        _dated_schedule_for(mid) for models in (regional or {}).values() for mid in models
    )
    _emit_shared_header(lines, has_dated=has_dated)
    lines.append("ANTHROPIC_PRICING: dict[str, ModelPricing] = {")
    for model_id in sorted(pricing):
        _emit_model_pricing(lines, model_id, pricing[model_id])
    lines.append("}")
    lines.append("")
    _emit_nested_pricing_dict(
        lines,
        "ANTHROPIC_REGIONAL_PRICING",
        regional,
        comment="Regional overrides for Claude (only Regions whose prices differ from us-east-1)",
    )
    lines.extend(_SHARED_ANTHROPIC_FUNCTION_BODY.splitlines())
    lines.append("")
    return "\n".join(lines)


def _dated_schedule_for(model_id: str) -> tuple[date, Decimal] | None:
    """Return (valid_from, multiplier) if ``model_id`` has a scheduled future price change."""
    for needle, schedule in DATED_PRICE_SCHEDULES.items():
        if needle in model_id:
            return schedule
    return None


def _scale_prices(mp: ModelPrices, multiplier: Decimal) -> ModelPrices:
    """Return a copy of ``mp`` with every non-None cost scaled by ``multiplier``."""
    scaled = {field.name: value * multiplier for field in fields(mp) if (value := getattr(mp, field.name)) is not None}
    return replace(mp, **scaled)


def _emit_tiers_block(lines: list[str], mp: ModelPrices, is_emb: bool, tiers_indent: int) -> None:
    """Emit a ``tiers=[...]`` block: the standard tier plus the long-context tier if present."""
    pad = " " * tiers_indent
    lines.append(f"{pad}tiers=[")
    _emit_tier(lines, mp, is_emb, tiers_indent + 4, is_lctx=False)
    if mp.has_lctx:
        _emit_tier(lines, mp, is_emb, tiers_indent + 4, is_lctx=True)
    lines.append(f"{pad}],")


def _emit_model_pricing(lines: list[str], model_id: str, mp: ModelPrices, indent: int = 4) -> None:
    """Emit a single ModelPricing entry, including a dated schedule override if one applies."""
    pad = " " * indent
    is_emb = _is_embedding(model_id)

    lines.append(f'{pad}"{model_id}": ModelPricing(')
    _emit_tiers_block(lines, mp, is_emb, indent + 4)

    schedule = _dated_schedule_for(model_id)
    if schedule is not None:
        valid_from, multiplier = schedule
        lines.append(f"{pad}    schedules=[")
        lines.append(f"{pad}        PricingSchedule(")
        lines.append(f"{pad}            valid_from=date({valid_from.year}, {valid_from.month}, {valid_from.day}),")
        _emit_tiers_block(lines, _scale_prices(mp, multiplier), is_emb, indent + 12)
        lines.append(f"{pad}        ),")
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
        cache_write_1h = mp.lctx_cache_write_1h_cost
    else:
        input_cost = mp.input_cost
        output_cost = mp.output_cost
        cache_read = mp.cache_read_cost
        cache_write = mp.cache_write_cost
        cache_write_1h = mp.cache_write_1h_cost

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
    if cache_write_1h is not None and cache_write_1h > 0:
        lines.append(
            f'{pad}    cache_creation_cost_per_token_by_ttl={{"1h": per_million_tokens({_fmt(cache_write_1h)})}},'
        )
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


def _get_price_with_unit(sku: str, terms: dict[str, Any]) -> tuple[Decimal, str] | None:
    """Extract the USD price and its billing unit from OnDemand terms for a SKU."""
    if sku not in terms:
        return None
    for offer in terms[sku].values():
        for dim in offer.get("priceDimensions", {}).values():
            usd = dim.get("pricePerUnit", {}).get("USD")
            if usd is not None:
                return Decimal(usd), dim.get("unit", "")
    return None


def _scale_to_per_million(price: Decimal, unit: str) -> Decimal | None:
    """Scale a token price to per-million based on its AWS billing unit.

    AWS labels token dimensions either ``1K tokens`` (the common case) or
    ``1M tokens`` (e.g. xAI Grok). Returns None for an unrecognized unit so the
    caller can skip the dimension rather than emit a 1000x-wrong price.
    """
    normalized = unit.strip().lower()
    if normalized in ("1k tokens", "1000 tokens"):
        return price * 1000
    if normalized in ("1m tokens", "1000000 tokens"):
        return price
    return None


def _warn(msg: str) -> None:
    """Print a warning to stderr."""
    print(f"WARNING: {msg}", file=sys.stderr)  # noqa: T201


def _die(msg: str) -> NoReturn:
    """Print an error to stderr and exit non-zero, leaving any existing generated files untouched."""
    print(f"ERROR: {msg}", file=sys.stderr)  # noqa: T201
    raise SystemExit(1)


def _region_overrides(  # noqa: PLR0913
    reg_bedrock: dict[str, Any],
    reg_fm: dict[str, Any],
    default_pricing: dict[str, ModelPrices],
    global_pricing: dict[str, ModelPrices],
    global_models: set[str],
    resolution_map: dict[str, str],
    region: str,
) -> dict[str, ModelPrices]:
    """A Region's standard (bare) and Global-profile (``global.``) overrides vs us-east-1.

    Standard overrides use only non-global meters (a Global-only meter is not a genuine standard
    rate). Global overrides come from the Global meters: a Global-profile call is billed by the
    Region it is called from, and AWS publishes a per-Region Global rate (uniform for Claude, but
    not for Nova/Titan), so those must be kept where they differ from the us-east-1 Global rate --
    keyed ``global.<id>`` -- rather than falling through to the single us-east-1 Global rate.

    ``global_models`` are the resolved model IDs that actually have a Global inference profile (the
    default table's ``global.`` keys). A Global override is emitted only for those: some models
    carry a Global *meter* in the price list but no invokable Global profile, so a ``global.`` key
    for them would price a call that cannot be made.
    """
    reg_mantle = parse_mantle_models(reg_bedrock)
    reg_amazon, reg_amazon_global = parse_amazon_models(reg_bedrock, fallback_to_global=False)
    reg_foundation, reg_foundation_global = parse_foundation_models(reg_fm, fallback_to_global=False)

    reg_std = resolve_pricing_ids(merge_pricing(reg_mantle, reg_amazon, reg_foundation), resolution_map)
    std_diffs, std_dropped = drop_unemittable(compute_regional_diffs(default_pricing, reg_std))

    reg_global = resolve_pricing_ids(merge_pricing({}, reg_amazon_global, reg_foundation_global), resolution_map)
    reg_global = {mid: mp for mid, mp in reg_global.items() if mid in global_models}
    global_diffs, global_dropped = drop_unemittable(compute_regional_diffs(global_pricing, reg_global))

    for model_id, reason in std_dropped:
        _warn(f"  {region}: {model_id} — {reason}; falling back to us-east-1 pricing")
    for model_id, reason in global_dropped:
        _warn(f"  {region}: global.{model_id} — {reason}; falling back to us-east-1 global pricing")

    return {**std_diffs, **{f"global.{model_id}": mp for model_id, mp in global_diffs.items()}}


def _fetch_regional_diffs(
    args: argparse.Namespace,
    default_pricing: dict[str, ModelPrices],
    global_pricing: dict[str, ModelPrices],
    global_models: set[str],
    resolution_map: dict[str, str],
) -> dict[str, dict[str, ModelPrices]] | None:
    """Fetch pricing for every Region (or those requested) and return overrides vs us-east-1.

    ``resolution_map`` must be the one already applied to ``default_pricing``/``global_pricing``:
    both sides of the comparison have to be keyed by real Bedrock model IDs, or every dated model
    misses the lookup and is reported as a spurious diff under an ID no request ever carries.
    ``global_models`` are the resolved IDs with a Global profile (see :func:`_region_overrides`).
    """
    if args.regions:
        # An explicit narrowing for quick partial runs. On a write it yields a partial table --
        # only the requested Regions keep overrides -- so warn rather than silently drop the rest.
        if args.write:
            _warn("--regions with --write emits overrides for only those Regions; omit --regions to refresh all.")
        regions = [r for r in args.regions if r != DEFAULT_REGION]
    else:
        # Default (including the documented bare --write): every Region, so a refresh regenerates
        # the full regional table instead of wiping it.
        bedrock_index = fetch_region_index("AmazonBedrock")
        fm_index = fetch_region_index("AmazonBedrockFoundationModels")
        regions = sorted((set(bedrock_index.keys()) | set(fm_index.keys())) - {DEFAULT_REGION})

    regional_diffs: dict[str, dict[str, ModelPrices]] = {}
    failed: list[str] = []
    for region in regions:
        _info(f"Fetching pricing for {region}...")
        try:
            reg_bedrock = fetch_pricing("AmazonBedrock", region)
            reg_fm = fetch_pricing("AmazonBedrockFoundationModels", region)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            _warn(f"Failed to fetch {region}: {e}")
            failed.append(region)
            continue

        overrides = _region_overrides(
            reg_bedrock, reg_fm, default_pricing, global_pricing, global_models, resolution_map, region
        )
        if overrides:
            regional_diffs[region] = overrides
            _info(f"  {region}: {len(overrides)} overrides differ from us-east-1")
        else:
            _info(f"  {region}: all prices match us-east-1")

    if failed:
        detail = ", ".join(failed)
        # A partial fetch must not overwrite a complete committed table: a skipped Region's genuine
        # overrides would vanish and every call there would silently fall back to us-east-1.
        if args.write:
            _die(
                f"{len(failed)} region(s) failed to fetch ({detail}); refusing to --write a table that would "
                "drop their overrides. Re-run when the Pricing API is reachable."
            )
        _warn(f"{len(failed)} region(s) failed to fetch ({detail}); the printed table omits them.")

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
        help="Only these regions get overrides (default: all regions)",
    )
    _ = parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Deprecated no-op: all regions are included by default",
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
    amazon, amazon_global = parse_amazon_models(bedrock_data, report_unmapped=True)
    _info(f"  Found {len(amazon)} Amazon/legacy models ({len(amazon_global)} with global pricing)")

    _info("Parsing Foundation Models (Claude/Cohere/etc)...")
    foundation, foundation_global = parse_foundation_models(fm_data)
    _info(f"  Found {len(foundation)} Foundation Models ({len(foundation_global)} with global pricing)")

    # Merge
    default_pricing = merge_pricing(mantle, amazon, foundation)
    global_pricing: dict[str, ModelPrices] = {**amazon_global, **foundation_global}
    _info(f"Total models after merge: {len(default_pricing)} ({len(global_pricing)} with global pricing)")

    # Fetch real model/profile IDs from Bedrock API and resolve pricing keys
    _info("Fetching Bedrock catalog...")
    real_model_ids, real_profile_ids = fetch_bedrock_catalog()
    _info(f"  Found {len(real_model_ids)} foundation models, {len(real_profile_ids)} inference profiles")

    all_simplified = set(default_pricing.keys()) | set(global_pricing.keys())
    resolution_map = build_id_resolution_map(all_simplified, real_model_ids)
    if resolution_map:
        default_pricing = resolve_pricing_ids(default_pricing, resolution_map)
        global_pricing = resolve_pricing_ids(global_pricing, resolution_map)

    expanded_pricing = expand_with_real_profiles(default_pricing, global_pricing, real_profile_ids)
    _info(f"Total entries after inference profile expansion: {len(expanded_pricing)}")

    # Regional pricing (compared against unexpanded default/global, re-keyed the same way). Only
    # models with a real Global profile (a "global." key in the expanded table) get Global overrides.
    global_models = {k[len("global.") :] for k in expanded_pricing if k.startswith("global.")}
    regional_diffs = _fetch_regional_diffs(args, default_pricing, global_pricing, global_models, resolution_map)

    # Split off the Anthropic-on-Bedrock subset (shared with the native lmux-anthropic Bedrock
    # provider), default and regional alike, so both consumers price Claude from the same table.
    # lmux-aws-bedrock merges the Anthropic subset back into its own tables.
    anthropic_pricing = {mid: mp for mid, mp in expanded_pricing.items() if _is_anthropic(mid)}
    bedrock_pricing = {mid: mp for mid, mp in expanded_pricing.items() if not _is_anthropic(mid)}
    anthropic_regional = _split_regional(regional_diffs, anthropic=True)
    bedrock_regional = _split_regional(regional_diffs, anthropic=False)

    # Generate code
    shared_code = generate_shared_anthropic_py(anthropic_pricing, anthropic_regional)
    bedrock_code = generate_cost_py(bedrock_pricing, bedrock_regional)

    if args.write:
        _ = SHARED_PRICING_PATH.write_text(shared_code)
        _info(f"Wrote {SHARED_PRICING_PATH}")
        _ = COST_PY_PATH.write_text(bedrock_code)
        _info(f"Wrote {COST_PY_PATH}")
    else:
        print(shared_code)  # noqa: T201
        print("\n\n# ===== lmux-aws-bedrock/cost.py =====\n")  # noqa: T201
        print(bedrock_code)  # noqa: T201


def _info(msg: str) -> None:
    """Print an info message to stderr."""
    print(msg, file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
