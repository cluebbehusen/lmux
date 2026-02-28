# AGENTS.md

## Project Overview

lmux is a modular Python library for unified LLM provider access with cost reporting. It's designed as a performance-first alternative to packages that are a kitchen sink of LLM functionality — separate packages per provider, lazy SDK loading, and no global state.

### Monorepo Structure

uv workspace with a virtual root (`package = false`). All packages live under `packages/`:

```text
packages/
├── lmux/                  # Core: types, protocols, exceptions, cost utils, mock provider
├── lmux-openai/           # OpenAI provider (implemented)
├── lmux-anthropic/        # Anthropic provider (skeleton)
├── lmux-azure-openai/     # Azure OpenAI provider (skeleton)
├── lmux-bedrock/          # AWS Bedrock provider (skeleton)
├── lmux-google/           # Google Gemini/Vertex provider (skeleton)
└── lmux-groq/             # Groq provider (skeleton)
```

Each package uses `src/` layout: `packages/<name>/src/<import_name>/` and `packages/<name>/tests/`.

Core (`lmux`) depends only on `pydantic`. Provider packages depend on `lmux` + their SDK (e.g., `lmux-openai` depends on `lmux` + `openai`).

## Architecture

### Protocols

Core defines four `@runtime_checkable` protocols in `protocols.py`:

- `AuthProvider[AuthT]` — sync/async credential retrieval
- `CompletionProvider[ParamsT]` — `chat`, `achat`, `chat_stream`, `achat_stream`
- `EmbeddingProvider[ParamsT]` — `embed`, `aembed`
- `ResponsesProvider[ParamsT]` — `create_response`, `acreate_response`

Providers implement only the protocols they support. `ParamsT` is a provider-specific Pydantic model (e.g., `OpenAIParams`). Auth is an implementation detail — not exposed in protocol method signatures.

### Provider Pattern

Every provider package follows the same structure (see lmux-openai as reference):

| File | Purpose |
| --- | --- |
| `provider.py` | Provider class implementing protocol(s) |
| `auth.py` | `<Provider>EnvAuthProvider` — reads API key from env var |
| `params.py` | `<Provider>Params(BaseModel)` — provider-specific parameters |
| `cost.py` | `PRICING` dict + `calculate_<provider>_cost()` with prefix matching |
| `_lazy.py` | `create_sync_client()` / `create_async_client()` factory functions |
| `_mappers.py` | Convert between lmux types and SDK types |
| `_exceptions.py` | Map SDK exceptions to `lmux.exceptions` hierarchy |
| `__init__.py` | Re-exports + `preload()` function |

### Lazy Loading

Provider SDKs load on first API call, not on import. The `_lazy.py` module contains factory functions that do `import <sdk>` inside the function body. Provider instances store their SDK client on the instance (not module-level globals). `preload()` is opt-in for eager initialization.

### Cost Ownership

Each provider owns its pricing data and cost calculation. Core provides `ModelPricing`, `calculate_token_cost()`, and `calculate_cost_from_usage()` utilities — no pricing database. Unknown models return `None` for cost, not an error. Providers use longest-prefix matching (e.g., `gpt-4o-2024-11-20` matches `gpt-4o`).

### Response Design

Flattened responses — no `choices` array. Always take the first choice. Every response carries `.cost` if the provider knows the pricing.

## Code Conventions

### Python Version & Syntax

- Python 3.12+ required
- PEP 695 generics: `class Foo[T]:`, `type Alias = ...`
- Union syntax: `X | Y`
- Pydantic v2 `BaseModel` for all types, params, and responses

### Type Annotations

- `TYPE_CHECKING` + string annotations for expensive SDK imports — **never** `from __future__ import annotations`
- `Sequence[Message]` (not `list[Message]`) in protocol signatures for covariance
- `# pyright: ignore[reportSpecificError]` — always include the specific error code
- `# noqa: CODE` — always include the specific rule code (e.g., `# noqa: A002`, `# noqa: PLR0913`)
- `# pragma: no cover` only for genuinely untestable code (lazy import stubs, `assert_never` branches)

### Naming

- Provider classes: `<Provider>Provider` (e.g., `OpenAIProvider`)
- Auth classes: `<Provider>EnvAuthProvider`
- Params classes: `<Provider>Params`
- Private/internal modules: prefixed with `_` (`_lazy.py`, `_mappers.py`, `_exceptions.py`)
- Parameter `input` is kept with `# noqa: A002` — matches LLM SDK convention

### Formatting

- Double quotes
- 120-character line length
- Spaces (not tabs)

## Tooling & Verification

All four checks must pass before finishing work:

```bash
# Lint
uv run ruff check

# Format
uv run ruff format --check

# Type check
uv run basedpyright 

# Tests with 100% branch coverage
uv run pytest
```

### Coverage

- `source` is configured to package names (e.g., `["lmux", "lmux_openai"]`) — update when adding new providers
- `branch = true` — branch coverage, not just line coverage
- `fail_under = 100`

## Testing Patterns

### General

- All tests are mocked unit tests — **no network calls**
- `unittest.mock`: `MagicMock`, `AsyncMock`, `patch`
- `pytest-asyncio` with `asyncio_mode = "auto"` — async test methods just work
- `--import-mode=importlib` to avoid namespace collisions between packages
- **No `tests/__init__.py` files** (required for importlib mode)
- 100% branch coverage required

### Test Structure

Test files mirror source modules, prefixed with `test_`:

```text
packages/lmux-openai/
├── src/lmux_openai/
│   ├── __init__.py
│   ├── provider.py
│   ├── auth.py
│   ├── params.py
│   ├── cost.py
│   ├── _lazy.py
│   ├── _mappers.py
│   └── _exceptions.py
└── tests/
    ├── test_provider.py
    ├── test_auth.py
    ├── test_params.py
    ├── test_cost.py
    ├── test_mappers.py
    └── test_exceptions.py
```

Each function under test gets its own class, and each test case is a method within that class:

```python
class TestMapMessages:
    def test_system_message(self) -> None:
        result = map_messages([SystemMessage(content="Be helpful.")])
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_user_message_text(self) -> None:
        result = map_messages([UserMessage(content="Hello")])
        assert result == [{"role": "user", "content": "Hello"}]
```

### Fixtures Over Helpers

Use **pytest fixtures** for all mocking and shared test data — not module-level helper functions. This includes:

- Mock clients and providers (e.g., `sync_provider`, `async_provider`)
- Reusable SDK response objects (e.g., `chat_completion`, `stream_chunks`, `embedding_response`)
- Fake auth providers
- Common test data (e.g., `api_error` factories)

Mock fixtures that use `patch()` are **path-dependent** (e.g., `patch("lmux_openai.provider.create_sync_client")`) — keep these in the individual test file, not `conftest.py`. Only truly shared, non-path-dependent data (e.g., reusable model instances, constants) belongs in `conftest.py`.

### Mocking Principles

- **One level deep only** (true unit testing). If `a` calls `b` which calls `c`, the test for `a` mocks `b` — never `c`. Each unit test exercises exactly one layer.
- **Type mocks correctly**: `MagicMock` for sync functions, `AsyncMock` for async functions.
- **Don't write test scaffolding that itself needs testing.** Keep test setup minimal — if a test file has large helper functions or complex builders, the test is testing its own scaffolding, not the library code.
- **Test for 100% branch coverage but don't over-test.** Cover all branches in *our* code. Don't write tests that effectively exercise third-party library behavior — that's what mocks are for.
- **Test positive and negative assertions** — verify that expected calls were made *and* that unexpected calls were not. This also avoids "unused argument" warnings from fixtures.
- Use `assert_called_once_with` / `assert_awaited_once_with` to verify mock calls — not `call_args`:

```python
def test_embed_calls_sdk(self, sync_provider: OpenAIProvider, mock_client: MagicMock) -> None:
    mock_client.embeddings.create.return_value = embedding_response
    sync_provider.embed("text-embedding-3-small", "hello")

    # Good — assert directly on the mock
    mock_client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input="hello")
    mock_client.chat.completions.create.assert_not_called()

    # Bad — don't inspect call_args directly
    call_kwargs = mock_client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-small"
```

### Data Assertions

Compare the whole object rather than asserting individual fields — this catches regressions if fields are added, dropped, or renamed:

```python
# Good — assert the entire object
assert result == ChatResponse(
    content="Hello!",
    tool_calls=None,
    usage=Usage(input_tokens=10, output_tokens=5),
    cost=Cost(input_cost=0.003, output_cost=0.0075, total_cost=0.0105),
    model="gpt-4o",
    provider="openai",
    finish_reason="stop",
)

# Bad — field-by-field assertions
assert result.content == "Hello!"
assert result.model == "gpt-4o"
assert result.provider == "openai"
assert result.usage.input_tokens == 10
```

### Provider Test Patterns

Tests mock at the SDK client level, not at the network level:

- Fixtures inject `MagicMock` as `_sync_client` / `AsyncMock` as `_async_client` on provider instances
- Client creation tests patch `<package>.provider.create_sync_client` (not `_lazy.create_sync_client`)
