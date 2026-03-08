---
name: update-pricing
description: Validate and update LLM provider pricing data by checking each provider's official pricing page. Use when pricing may be stale or when new models have been released.
---

# Update Pricing

Validate existing pricing data and discover new models by checking each provider's official pricing page.

## Workflow

### Step 1: Discover providers

Find all provider `cost.py` files by globbing for `packages/lmux-*/src/*/cost.py`.
For each file found:

1. Extract the **provider name** from the path (e.g., `packages/lmux-openai/...` → `openai`)
2. Read the module docstring and extract the **pricing source URL** from the
   `Pricing source:` line. If a cost.py has no `Pricing source:` line, skip
   it and warn the user.

Build a list of `(provider_name, cost_py_path, pricing_url)` tuples.

### Step 2: Select providers

Create the scratch output directory:

```
tmp/update-pricing/YYYY-MM-DD/
```

Use today's date. If the directory already exists from a prior run, files will
be overwritten.

Present a multi-select form to the user using `AskUserQuestion` listing all
discovered providers. All providers should be selected by default. The user can
deselect providers they don't want to check.

### Step 3: Launch parallel subagents

Spin up one Agent per selected provider **in a single message** so they run in
parallel. If possible, spin up each subagent IN THE BACKGROUND.
Each agent MUST receive the following in its prompt:

1. The provider name
2. The path to the provider's `cost.py` file (the subagent will read it itself)
3. The pricing source URL
4. The complete "Subagent Instructions" section below (copy it verbatim)
5. The complete "Report Format" section below (copy it verbatim)
6. The scratch output path: `tmp/update-pricing/YYYY-MM-DD/<provider>.md`

Each agent MUST:

1. Read the provider's `cost.py` file
2. Use the web to retrieve the pricing page
3. Compare every model and price in the cost.py against the source
4. Write a detailed findings report to the scratch file
5. Return ONLY a single-line summary (no other output)

### Step 4: Review findings

After ALL agents complete, read each `tmp/update-pricing/YYYY-MM-DD/<provider>.md`
file. Present a consolidated summary to the user showing:

- Which providers have discrepancies
- Which providers have new models available
- Which providers are fully verified
- Any caveats or fetch failures

### Step 5: Apply updates (only if user approves)

Do NOT automatically apply changes. Ask the user whether to proceed. If
approved:

1. Update each provider's `cost.py` with corrected prices and new models
2. Follow existing code patterns exactly:
   - Use `per_million_tokens()` for all per-token prices
   - Maintain alphabetical grouping by model family with comment headers
   - Maintain the `_PRICING_BY_PREFIX` sorted list after `_PRICING`
   - Keep multi-tier pricing for providers that use it (anthropic, gcp-vertex)
   - Keep `cache_creation_cost_per_token` for anthropic models
   - Preserve multiplier constants and `apply_cost_multiplier()` functions
3. Run verification per AGENTS.md
4. Fix any issues and re-run verification until all four checks pass

---

## Large pricing pages

Some pricing pages (notably GCP Vertex AI) are too large for `WebFetch` to
return in full — it truncates before reaching partner/embedding model sections.

For these pages, subagents should use `curl` via Bash to download the full HTML
to a temp file, then extract relevant sections with `sed`/`grep`/`python`. For
example:

```bash
curl -s 'https://cloud.google.com/vertex-ai/generative-ai/pricing' > /tmp/vertex-pricing.html
# Extract partner models section (Claude, Mistral, Llama, etc.)
sed -n '28100,28800p' /tmp/vertex-pricing.html | python3 -c "
import sys, html, re
content = sys.stdin.read()
content = re.sub(r'<tr[^>]*>', '\n|ROW|', content)
content = re.sub(r'<td[^>]*>', '|', content)
content = re.sub(r'<[^>]+>', '', content)
content = html.unescape(content)
for line in [l.strip() for l in content.split('\n') if l.strip()]:
    print(line)
"
```

Line ranges shift as the page is updated — use `grep -n` to find section
anchors (e.g., `id="partner-models"`, `id="embedding-models"`) first.

---

## Subagent Instructions

You are a pricing validation subagent for the **{provider}** provider.

### Your task

1. Read the provider's `cost.py` file at `{cost_py_path}`. Study the `_PRICING`
   dict carefully — note every model key and its pricing fields.
2. Use the web to fetch the pricing page: `{pricing_url}`.
   - If the page fails to load or returns an error, note this in the Caveats
     section and set the report status to `FETCH_FAILED`. List all models from
     cost.py as "unverifiable".
   - If the page loads but pricing data is incomplete or hard to parse (e.g.,
     requires JavaScript rendering), note this in Caveats and set status to
     `PARTIAL_VERIFICATION`.
3. For each model in `_PRICING`:
   - Find the corresponding model on the pricing page
   - Compare: input cost, output cost, cache read cost, cache creation cost
   - If any value differs, add a row to the "Price Updates" table
   - If the model cannot be found on the pricing page, note it in "Caveats"
     as potentially deprecated or renamed
4. Scan the pricing page for models NOT in `_PRICING`:
   - Add any new models to the "New Models" table
   - Only include models relevant to this provider package
5. Check for any pricing complexities (regional differences, deployment-type
   multipliers, tiered pricing) that the code doesn't handle. Report these
   in "Caveats".
6. Write your findings to the scratch file path provided using the exact
   report format below.
7. Return ONLY a single-line summary in one of these formats:
   - `"openai: 3 price updates, 2 new models"`
   - `"anthropic: all 14 models verified"`
   - `"aws-bedrock: FETCH_FAILED — could not retrieve pricing page"`

### Thoroughness

Be extremely thorough. LLM pricing is complex — it can vary by region,
deployment type, caching strategy, token volume tiers, and more. Study the
provider code files carefully to understand what the code currently handles (e.g.,
multiplier constants, multi-tier pricing, cache read/write costs) and verify
ALL of it against the pricing page. If the pricing page reveals complexity
that the code doesn't account for, report it in Caveats.

---

## Report Format

Write findings to `tmp/update-pricing/YYYY-MM-DD/{provider}.md` using exactly
this format:

```markdown
# Status: {DISCREPANCIES_FOUND | ALL_VERIFIED | FETCH_FAILED | PARTIAL_VERIFICATION}

# Provider: {provider-name}

# Pricing source: {url}

# Date: {YYYY-MM-DD}

# Models checked: {N}

# Models in cost.py: {M}

## Price Updates

| Model           | Tier  | Field       | Current (per M) | Actual (per M) |
| --------------- | ----- | ----------- | --------------- | -------------- |
| gpt-4o          | base  | output_cost | 10.0            | 7.5            |
| claude-sonnet-4 | >200k | input_cost  | 6.0             | 5.5            |

(If no updates needed, write "No price discrepancies found.")

## New Models

| Model            | Input (per M) | Output (per M) | Cache Read (per M) | Cache Write (per M) | Notes                   |
| ---------------- | ------------- | -------------- | ------------------ | ------------------- | ----------------------- |
| gpt-4o-audio     | 2.50          | 10.00          | ---                | ---                 |                         |
| claude-haiku-4-6 | 0.80          | 4.00           | 0.08               | 1.00                | Multi-tier; see caveats |

(If no new models, write "No new models found.")

## Caveats

- Any regional pricing complexities not handled in code
- Models in cost.py not found on the pricing page (possible deprecation)
- Multi-tier pricing details for new models
- Data quality issues (page didn't load, pricing behind JS rendering, etc.)
- Multiplier constant changes

(If no caveats, write "None.")

## Verified (no changes needed)

gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, ...
```

### Tier column values

- Single-tier models: `base`
- Multi-tier models: `base` for the first tier, `>Nk` for higher tiers
  (e.g., `>200k`, `>128k`)
