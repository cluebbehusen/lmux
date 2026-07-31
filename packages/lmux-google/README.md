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

## Provider Params

```python
from lmux_google import GoogleParams

response = provider.chat(
    "gemini-2.5-pro",
    messages,
    provider_params=GoogleParams(thinking_config={"thinking_budget": 1024}),
)
```

| Parameter           | Type                  | Description                      |
| ------------------- | --------------------- | -------------------------------- |
| `safety_settings`   | `list[SafetySetting]` | Content safety thresholds        |
| `presence_penalty`  | `float`               | Presence penalty                 |
| `frequency_penalty` | `float`               | Frequency penalty                |
| `seed`              | `int`                 | Deterministic sampling seed      |
| `labels`            | `dict[str, str]`      | Request labels                   |
| `thinking_config`   | `dict`                | Thinking/reasoning configuration |

## Constructor Options

```python
GoogleProvider(
    auth=...,       # AuthProvider, default: GoogleADCAuthProvider()
    project=...,    # GCP project ID
    location=...,   # GCP region
    vertexai=...,   # Use Vertex AI (default: True) vs. AI Studio
    timeout=...,    # request timeout in seconds
    max_retries=..., # retry count for transient failures
    transport=...,        # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,  # Optional httpx.AsyncBaseTransport for the async client
)
```
