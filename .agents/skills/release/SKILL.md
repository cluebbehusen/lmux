---
name: release
description: Cut a release for an lmux package — create the GitHub release whose tag triggers the PyPI publish, with hand-written notes in the repo's changelog style. Use ONLY when a maintainer explicitly asks to release or publish a package; never on your own initiative.
---

# Release

Cut a release for one or more lmux packages: create a GitHub release whose git tag triggers the publish workflow, with release notes written in this repo's conventional-commit-derived style.

> **Only run this when a maintainer explicitly asks to cut or publish a release.** Publishing is outward-facing and effectively irreversible (a PyPI version cannot be replaced once uploaded), so it is never done on your own initiative and never as an automatic follow-on to merging a PR. If in doubt, stop and ask.

## How releases work here

- Packages are versioned and released **independently** (`packages/lmux`, `packages/lmux-anthropic`, ...). One release == one package. A change that bumps several packages needs one release per package.
- `.github/workflows/publish.yml` triggers on a pushed git tag matching `lmux-v*` or `lmux-<pkg>-v*` and publishes that package to PyPI via Trusted Publishing. It derives the package from the tag, reads that package's `pyproject.toml` version, and **fails if the tag version does not match the pyproject version** — so the version must already be bumped and merged.
- `gh release create <tag>` creates the tag at the target commit *and* the GitHub release in one step. The pushed tag is what triggers publish, so creating the release **is** the publish action. There is no separate `git tag` / `git push --tags`.
- There is no changelog automation. Notes are written by hand in the format below.
- (`workflow_dispatch` on the publish workflow can target TestPyPI or re-run a publish manually; the normal path is a tag/release.)

## Tag and title format

- **Tag** (dashed; this is what the workflow matches): `lmux-v<version>` for the core package, `lmux-<pkg>-v<version>` for providers. E.g. `lmux-v0.9.0`, `lmux-anthropic-v0.10.0`.
- **Title** (spaced; human-readable): `<package> v<version>`. E.g. `lmux v0.9.0`, `lmux-anthropic v0.10.0`.
- `<version>` MUST equal the package's current `pyproject.toml` version.

## Notes format

Group entries by type; include only user-facing changes:

- `### Features` — from `feat:` commits
- `### Bug Fixes` — from `fix:` commits
- `### Deprecation` — when a package or name is being deprecated (rare)

Each entry is `- **<scope>**: <what changed and its user-facing impact>.` — a short bolded scope (the area, e.g. `pricing`, `reasoning`, `vertex`, `responses`, `foundry`), then one or two complete sentences describing the change concretely: name the new params, prices, models, dates, or behavior. Omit non-user-facing commits (`style`, `chore`, `test`, `docs`, `ci`, and refactors with no behavior change). A trivial release (e.g. a rename shim) can use a one-line body instead of sections.

Example (matches existing releases):

```markdown
### Features

- **pricing**: Add Claude Sonnet 5 with date-based pricing — introductory rates through 2026-08-31, standard from 2026-09-01. Live costs reflect the rate in effect on the request date; pass `AnthropicParams(pricing_as_of=...)` to bill against a specific date.

### Bug Fixes

- **pricing**: Cross-region inference-profile model ids without a dedicated entry now fall back to the base model's pricing instead of returning no cost.
```

## Workflow

### Step 1: Confirm the ask and the base

Confirm the maintainer asked for this release and which package(s) and version(s). Releases are cut from `main` after the change has merged. Update main and confirm the pyproject version is the one to publish:

```bash
git fetch origin && git switch main && git pull
grep '^version' packages/<pkg>/pyproject.toml
```

### Step 2: Verify the package is green

Run the full gate from AGENTS.md and do not release a package that isn't green:

```bash
uv run ruff format --check && uv run ruff check && uv run basedpyright && uv run pytest
```

### Step 3: Draft the notes and get sign-off

For each package, list its commits since its last release and turn the user-facing ones into notes:

```bash
git log lmux-<pkg>-v<last-version>..HEAD -- packages/lmux-<pkg>   # core: packages/lmux
```

Assign a scope to each and write impact-focused sentences (see format above). Present the drafted **title + notes** for each package to the maintainer and get explicit sign-off before creating anything.

### Step 4: Create the release(s)

Once notes are approved, for each package (put the notes in a file to avoid shell-escaping):

```bash
gh release create lmux-<pkg>-v<version> \
  --title "lmux-<pkg> v<version>" \
  --target main \
  --notes-file <notes-file>
# core package:
gh release create lmux-v<version> --title "lmux v<version>" --target main --notes-file <notes-file>
```

This tags `main` and triggers `publish.yml` for that package. When releasing the core package alongside providers, release core first so providers resolving the new core version have it available.

### Step 5: Confirm the publish

Watch the publish workflow and report the result:

```bash
gh run watch "$(gh run list --workflow=publish.yml -L1 --json databaseId -q '.[0].databaseId')"
```

Report which packages published and at what versions. If a publish fails (commonly a tag/pyproject version mismatch), surface the error rather than retrying blindly.
