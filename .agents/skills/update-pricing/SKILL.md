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

`lmux-gcp-vertex` is a deprecated shim that re-exports its cost calculation
from `lmux-google` — it has no pricing of its own and should **never** be
covered. Silently exclude it from discovery (no warning); its pricing is
already validated via `lmux-google`.

Build a list of `(provider_name, cost_py_path, pricing_url)` tuples.

### Step 2: Select providers and reserve an isolated run directory

Present a multi-select form to the user using `AskUserQuestion` listing all
discovered providers. All providers should be selected by default. The user can
deselect providers they don't want to check.

After provider selection, determine a stable lowercase, kebab-case
`runner_slug` for the coding agent coordinating the run:

- Codex: `codex`
- Claude Code: `claude`
- Other coding agents: their short product name (for example, `cursor`)

Use the coordinating agent's identity, not a provider subagent's identity. All
subagents launched by one invocation MUST share the same runner slug and run
directory.

Capture the local date and time once at the start of the run as `RUN_DATE`
(`YYYY-MM-DD`) and `RUN_TIME` (`HHMMSS`). Do not recalculate either value
later, including if the run crosses midnight.

Atomically reserve a new scratch directory under:

```
tmp/update-pricing/<RUN_DATE>/<runner-slug>/<RUN_TIME>/
```

For example, concurrent runs may use
`tmp/update-pricing/2026-07-15/codex/091145/` and
`tmp/update-pricing/2026-07-15/claude/091146/`. Create the date and runner
parents if needed, then reserve the time directory with an atomic create. If
that exact time directory already exists, NEVER reuse or overwrite it.
Atomically try `<RUN_TIME>-2`, then `<RUN_TIME>-3`, and so on until
directory creation succeeds. Do not use a separate existence check followed by
creation, because concurrent invocations of the same coding agent can start in
the same second. Only parent creation may use `mkdir -p`; reserve each
candidate time directory with plain `mkdir` so the create itself is the
collision check.

Call the successfully reserved path `RUN_DIR` and use it unchanged for the
rest of the workflow.

### Step 3: Launch parallel subagents

Spin up one Agent per selected provider **in a single message** so they run in
parallel. If possible, spin up each subagent IN THE BACKGROUND.
Each agent MUST receive the following in its prompt:

1. The provider name
2. The path to the provider's `cost.py` file (the subagent will read it itself)
3. The pricing source URL
4. The complete "Subagent Instructions" section below (copy it verbatim)
5. The complete "Report Format" section below (copy it verbatim)
6. The runner slug, `RUN_DATE`, `RUN_TIME`, `RUN_DIR`, exact report path
   `<RUN_DIR>/<provider>.md`, and provider work directory
   `<RUN_DIR>/work/<provider>/` (call this `WORK_DIR`)
7. If the provider has a subsection under "Provider-specific instructions"
   below, copy that subsection verbatim too — it overrides the generic
   instructions where they conflict

Each agent MUST:

1. Read the provider's `cost.py` file
2. Use the web to retrieve the pricing page
3. Compare every model and price in the cost.py against the source
4. Create its assigned provider work directory, then keep every downloaded
   page, browser capture, API response, generated diff, stdout/stderr capture,
   and other intermediate artifact inside it
5. Write a detailed findings report to the exact report path
6. Return ONLY a single-line summary (no other output)

Never assign the same report path or `WORK_DIR` to two live subagents. If a
provider must be retried, wait for or stop the first attempt, then allocate
new `<provider>-retry-N.md` and `work/<provider>-retry-N/` paths. Preserve
the first attempt for diagnosis and record which retry report is authoritative.

### Step 4: Review findings

After ALL agents complete, read each `<RUN_DIR>/<provider>.md` file. Do not
discover reports by taking the newest date directory or globbing across sibling
runner directories; another coding agent may be writing there concurrently.
Present a consolidated summary to the user, including the exact `RUN_DIR`,
showing:

- Which providers have discrepancies
- Which providers have new models available
- Which providers are fully verified
- Any caveats or fetch failures

### Step 5: Apply updates (only if user approves)

Do NOT automatically apply changes. Ask the user whether to proceed. If
approved:

1. Re-read every target `cost.py` and confirm it is identical to the version
   the corresponding subagent audited. Also confirm repository `HEAD` has not
   moved. If either changed, STOP and reconcile the newer state before editing.
   Never apply a report against stale source.
2. Do not enter this mutation phase concurrently with another pricing run in
   the same worktree. Audits may run concurrently because their `RUN_DIR`
   values are isolated; tracked-file updates may not.
3. Update each provider's `cost.py` with corrected prices and new models
4. Follow existing code patterns exactly:
   - Use `per_million_tokens()` for all per-token prices
   - Maintain alphabetical grouping by model family with comment headers
   - Maintain the `_PRICING_BY_PREFIX` sorted list after `_PRICING`
   - Keep multi-tier pricing for providers that use it (anthropic, google)
   - Keep `cache_creation_cost_per_token` for anthropic models
   - Preserve multiplier constants and `apply_cost_multiplier()` functions
5. Run verification per AGENTS.md
6. Fix any issues and re-run verification until all four checks pass

---

## Large pricing pages

Some pricing pages (notably GCP Vertex AI) are too large for `WebFetch` to
return in full — it truncates before reaching partner/embedding model sections.

For these pages, subagents should use `curl` via Bash to download the full HTML
inside their assigned `WORK_DIR`, then extract relevant sections with
`sed`/`grep`/`python`. Never use a shared path in `/tmp` or the repository
root. For example:

```bash
curl -s 'https://cloud.google.com/vertex-ai/generative-ai/pricing' > "{WORK_DIR}/vertex-pricing.html"
# Extract partner models section (Claude, Mistral, Llama, etc.)
sed -n '28100,28800p' "{WORK_DIR}/vertex-pricing.html" | python3 -c "
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

## JavaScript-rendered pages (Playwright CLI)

Some pricing pages (notably **Azure**, and the `openai.com/api/pricing` shell)
render prices client-side, so `WebFetch` and `curl` return an empty shell with
**no numbers** — the point where a subagent otherwise falls back to unreliable
third-party aggregators. Such pages also often **hide rows behind UI toggles**
(e.g. OpenAI's Short/Long-context switch, or Standard/Batch/Priority service-tier
tabs). When curl/WebFetch yield no prices — or an obviously incomplete subset —
escalate to a real browser via the **Playwright CLI**, if available.

Check availability first: `playwright-cli --version` (or `npx --no-install
playwright cli --version`). If neither works, note it in Caveats and treat any
aggregator figures as low-confidence.

Prefer **structured data over scraping pixels**, in this order:

1. **A first-party pricing API**, if the provider has one — the most reliable
   source, no browser needed. Azure exposes the unauthenticated **Azure Retail
   Prices API** (see the `azure-foundry` provider notes).
2. **The page's own JSON.** Many docs sites embed pricing in a `<script>` blob or
   fetch it via XHR. With the page open, `playwright-cli requests` lists network
   calls — capture the pricing JSON directly rather than reading the DOM.
3. **The rendered DOM, extracted as data.** Open the page, reveal any hidden rows
   (click context-length / service-tier toggles), then pull whole tables as JSON:

   ```bash
   playwright-cli open '<pricing_url>'
   # click any toggle that reveals more columns/rows first, e.g.:
   #   playwright-cli find "Long context"; playwright-cli click <ref>
   playwright-cli --raw eval "JSON.stringify([...document.querySelectorAll('table')].map(t=>({head:[...t.querySelectorAll('tr:first-child th')].map(c=>c.textContent.trim()),rows:[...t.querySelectorAll('tbody tr')].map(r=>[...r.querySelectorAll('th,td')].map(c=>c.textContent.trim()))})))" > "{WORK_DIR}/tables.json"
   playwright-cli close
   ```

   Watch for **multi-level headers** — e.g. OpenAI's `Short context` / `Long
context` bands each span their own Input/Cached/Cache-write/Output columns, so
   one row carries both the base tier and the `>272k` tier. And there are usually
   **separate Standard / Batch / Priority tables**; only the **Standard on-demand**
   table maps to `cost.py`.

Note: `playwright-cli --raw eval` may **double-encode** JSON (returns a quoted
string). If `json.loads(...)` yields a `str`, decode it a second time.

---

## Provider-specific instructions

### aws-bedrock

Bedrock pricing is generated by `scripts/update_bedrock_pricing.py` from the
AWS Pricing API — do NOT scrape pricing pages. The generator emits **two**
files: the Anthropic-on-Bedrock subset to
`packages/lmux-bedrock-shared/src/lmux_bedrock_shared/pricing.py` (shared with
the native `lmux-anthropic` Bedrock provider) and every other vendor to
`packages/lmux-aws-bedrock/src/lmux_aws_bedrock/cost.py`, which merges the
shared Anthropic table back into its `_PRICING` via `**ANTHROPIC_PRICING`.
Discovery only globs `cost.py`, so validate both through the aws-bedrock
provider — `lmux_aws_bedrock.cost._PRICING` still contains every model
(Anthropic included) after the merge. Steps:

1. Run the script without `--write` (it prints both generated files to stdout,
   separated by a `# ===== lmux-aws-bedrock/cost.py =====` marker). It requires
   AWS credentials — e.g. `AWS_PROFILE=dev` after an SSO login; if credentials
   are missing, report `FETCH_FAILED` and tell the user to log in.
2. Diff the generated output against the current `pricing.py` and `cost.py` and
   report the differences as Price Updates / New Models.
3. ALWAYS check the script's stderr for `Unmapped Foundation Models
servicenames` warnings — models missing from `FM_SERVICENAME_MAP` are
   silently dropped from the output. Report each unmapped model as a New
   Model with a caveat that it needs an `FM_SERVICENAME_MAP` entry. A clean
   diff does NOT mean there are no new models.

If updates are approved (Step 5), add any missing `FM_SERVICENAME_MAP`
entries, then regenerate with `--write` (it rewrites both files) rather than
hand-editing them.

### azure-foundry

Azure's pricing pages are heavily JavaScript-rendered — `WebFetch`/`curl` return
an empty shell. Do **NOT** rely on third-party aggregators (they frequently
transpose input/output or conflate SKU versions). Use Azure's first-party sources:

1. **Azure Retail Prices API** (no auth, authoritative) — the best source for
   exact per-token meters. Filter by OData; page through with `$top`:

   ```bash
   curl -s "https://prices.azure.com/api/retail/prices?\$filter=contains(meterName,'embedding')&\$top=1000"
   ```

   `cost.py` stores the **Global Standard** base rate — the `-glbl` meters (e.g.
   `text-embedding-3-large-glbl`). The `-regnl` and `-Dzone` meters are the +10%
   regional / data-zone variants already applied separately via the
   `REGIONAL_MULTIPLIER` / `DATA_ZONE_MULTIPLIER` constants, so do NOT bake that
   premium into the base — a base priced at 1.1× the `-glbl` rate is a
   double-counting bug. (Some models live under a `Managed Model Hosting Service`
   product rather than `Azure Llama Models`; match by the model name in
   `meterName`, not the product.)

2. **Playwright CLI** to render the DeepSeek / Llama / Grok / Mistral Foundry
   pricing tables when you need the on-page layout — see "JavaScript-rendered
   pages (Playwright CLI)".
3. **Microsoft Learn** for lifecycle — check the **Retired Foundry Models** and
   model-retirement-schedule pages to catch models that should be _removed_ from
   `cost.py`. Absence from the pricing catalog alone is NOT proof of retirement;
   the retired-models page is.

### anthropic

Claude is sold both directly (Anthropic API) and on GCP Vertex AI, and the
two are expected to match: Vertex bills Anthropic list prices on its global
endpoint. Check BOTH sources:

1. The Anthropic pricing page (the `Pricing source:` URL from cost.py)
2. The Claude section of the Vertex AI pricing page:
   https://cloud.google.com/vertex-ai/generative-ai/pricing

Compare cost.py against both. A mismatch between the two pages' global
prices is itself a finding — flag it in Caveats even if cost.py agrees with
one of them. Also verify `VERTEX_REGIONAL_MULTIPLIER` (the regional and
multi-region endpoint premium) and the `_VERTEX_PREMIUM_PRICING_MODELS` /
`_VERTEX_UNIFORM_PRICING_MODELS` scope lists against the Vertex page's
per-endpoint tables.

Claude in Microsoft Foundry bills Anthropic's standard API pricing, so it
needs no separate pricing check — but if the Anthropic docs start describing
Foundry-specific deployment types or premiums, flag that in Caveats.

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
   - If the page is JavaScript-rendered and curl/WebFetch return no prices (or an
     incomplete subset), do NOT jump to third-party aggregators. First escalate
     to a first-party pricing API and/or the **Playwright CLI** — see
     "JavaScript-rendered pages (Playwright CLI)". Only if all first-party routes
     fail, fall back to aggregators, mark those figures low-confidence in Caveats,
     and set status to `PARTIAL_VERIFICATION`.
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
6. Keep every local intermediate artifact in the assigned `WORK_DIR`. Do not
   write generic filenames in the repository root or shared `/tmp`, and do
   not read artifacts from a sibling runner directory.
7. Write your findings only to the exact report path provided. Do not select or
   reuse a sibling runner directory. Use the exact report format below.
8. Return ONLY a single-line summary in one of these formats:
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

Write findings to `{RUN_DIR}/{provider}.md` using exactly this format:

```markdown
# Status: {DISCREPANCIES_FOUND | ALL_VERIFIED | FETCH_FAILED | PARTIAL_VERIFICATION}

# Provider: {provider-name}

# Runner: {runner-slug}

# Run directory: {RUN_DIR}

# Pricing source: {url}

# Date: {RUN_DATE}

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
