# lmux-google

Google (Gemini) provider for [lmux](https://github.com/cluebbehusen/lmux). Talks to the Google Gemini REST API directly over [httpx](https://pypi.org/project/httpx/), using [google-auth](https://pypi.org/project/google-auth/) to resolve Vertex AI credentials. Serves Google's models through either backend:

- **Vertex AI** (default) — authenticated with Google Cloud credentials
- **Gemini Developer API** (AI Studio) — authenticated with an API key (`vertexai=False`)

Supports chat completions, streaming, and embeddings for Google-published models: Gemini and the Gemini/text embedding models.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Auth

Three authentication methods:

### Application Default Credentials (default)

Uses `google.auth.default()`, which works with `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud` CLI, or instance metadata.

```python
from lmux_google import GoogleProvider

provider = GoogleProvider(project="my-project", location="us-central1")
```

### Service Account

```python
from lmux_google import GoogleServiceAccountAuthProvider

provider = GoogleProvider(
    project="my-project",
    location="us-central1",
    auth=GoogleServiceAccountAuthProvider(service_account_file="/path/to/key.json"),
)
```

### API Key

Set `GOOGLE_API_KEY` in your environment:

```python
from lmux_google import GoogleAPIKeyAuthProvider

provider = GoogleProvider(auth=GoogleAPIKeyAuthProvider(), vertexai=False)
```

## Usage

### Chat

```python
from lmux import UserMessage

response = provider.chat("gemini-2.5-pro", [UserMessage(content="Hello")])
print(response.content)
print(response.cost)
```

### Streaming

```python
for chunk in provider.chat_stream("gemini-2.5-pro", [UserMessage(content="Hello")]):
    if chunk.delta:
        print(chunk.delta, end="")
```

### Tool continuations

Gemini 3 models require their thought signatures on follow-up tool calls. `lmux-google` captures the native assistant parts in `response.continuation`; use `to_assistant_message()` to preserve them:

```python
response = provider.chat(model, messages, tools=tools)
messages.append(response.to_assistant_message())
messages.append(ToolMessage(content=tool_result, tool_call_id=response.tool_calls[0].id))
```

The provider replays a matching Google continuation exactly. If no matching continuation is present, it builds the assistant turn from normalized content and tool calls as before.

### Embeddings

```python
response = provider.embed("text-embedding-005", "Hello")
print(response.embeddings)
```

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`.

### Registry

Use with the lmux registry to route across multiple providers:

```python
from lmux import Registry

registry = Registry()
registry.register("google", provider)
response = registry.chat("google/gemini-2.5-pro", messages)
```

## Reasoning

The unified `reasoning_effort` parameter maps `low`, `medium`, and `high` to the native control supported by each
Gemini generation. Gemini 2.5 uses numeric `thinkingBudget` values: `1_024`, `8_192`, and `24_576` for Flash and
Flash-Lite, with `32_768` for Pro at high effort. Gemini 3 and later use `thinkingLevel` values `LOW`, `MEDIUM`, and
`HIGH`. All mappings request thought summaries with `includeThoughts: true`.

Model aliases that do not identify their generation cannot be mapped safely. For those aliases, pass an explicit
native `thinking_config` as shown below.

## Provider Params

```python
from lmux_google import GoogleParams

response = provider.chat(
    "gemini-2.5-pro",
    messages,
    provider_params=GoogleParams(thinking_config={"thinkingBudget": 1024, "includeThoughts": True}),
)
```

`thinking_config` is passed through verbatim using the native REST field names and takes precedence over the
top-level `reasoning_effort` parameter.

| Parameter           | Type                  | Description                                                                                                      |
| ------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `safety_settings`   | `list[SafetySetting]` | Content safety thresholds                                                                                        |
| `presence_penalty`  | `float`               | Presence penalty                                                                                                 |
| `frequency_penalty` | `float`               | Frequency penalty                                                                                                |
| `seed`              | `int`                 | Deterministic sampling seed                                                                                      |
| `labels`            | `dict[str, str]`      | Request labels                                                                                                   |
| `thinking_config`   | `dict`                | Thinking/reasoning configuration                                                                                 |
| `task_type`         | `str`                 | Embedding task type; not all embedding models make use of this when provided                                     |
| `pricing_as_of`     | `datetime.date`       | Override the date used for dated pricing (e.g. a model's introductory-rate window); defaults to the current date |

## Constructor Options

```python
GoogleProvider(
    auth=...,       # AuthProvider, default: GoogleADCAuthProvider()
    project=...,    # GCP project ID
    location=...,   # GCP region
    vertexai=...,   # Use Vertex AI (default: True) vs. AI Studio
    timeout=...,    # request timeout in seconds
    max_retries=..., # retry count for transient failures
    default_headers=...,  # Optional headers included with every request
    transport=...,        # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,  # Optional httpx.AsyncBaseTransport for the async client
)
```

`default_headers` is useful for gateway authentication, tracing, and routing. Google-managed authentication,
quota-project, and content-type headers take precedence over caller values, case-insensitively.

## Pricing

Rates are Vertex global-endpoint list prices. Setting `location` to anything other than `global` puts the
request on a non-global Vertex endpoint, which bills a 10% premium on the models Vertex publishes a
non-global rate for; the provider applies that automatically via `VERTEX_NON_GLOBAL_MULTIPLIER`. Every other
model bills list price on both endpoints, and the Gemini Developer API (`vertexai=False`) has no endpoint
premium at all.

Models with time-boxed introductory rates carry dated schedules, so cost reflects the rate in effect on the
request date. Pass `GoogleParams(pricing_as_of=...)` to price against a different date. The endpoint premium
is dated too — it starts at `VERTEX_NON_GLOBAL_PREMIUM_START` (2026-07-01), so costs replayed against an
earlier date take no multiplier.
