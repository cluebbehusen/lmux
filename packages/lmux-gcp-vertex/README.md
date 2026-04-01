# lmux-gcp-vertex

Google Cloud Vertex AI provider for [lmux](https://github.com/cluebbehusen/lmux). Wraps the [google-genai](https://pypi.org/project/google-genai/) SDK.

Supports chat completions, streaming, and embeddings.

## Auth

Three authentication methods:

### Application Default Credentials (default)

Uses `google.auth.default()`, which works with `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud` CLI, or instance metadata.

```python
from lmux_gcp_vertex import GCPVertexProvider

provider = GCPVertexProvider(project="my-project", location="us-central1")
```

### Service Account

```python
from lmux_gcp_vertex import GCPVertexServiceAccountAuthProvider

provider = GCPVertexProvider(
    project="my-project",
    location="us-central1",
    auth=GCPVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json"),
)
```

### API Key

Set `GOOGLE_API_KEY` in your environment:

```python
from lmux_gcp_vertex import GCPVertexAPIKeyAuthProvider

provider = GCPVertexProvider(auth=GCPVertexAPIKeyAuthProvider(), vertexai=False)
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

### Embeddings

```python
response = provider.embed("text-embedding-005", "Hello")
print(response.embeddings)
```

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`.

## Provider Params

```python
from lmux_gcp_vertex import GCPVertexParams

response = provider.chat(
    "gemini-2.5-pro",
    messages,
    provider_params=GCPVertexParams(thinking_config={"thinking_budget": 1024}),
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
GCPVertexProvider(
    auth=...,       # AuthProvider, default: GCPVertexADCAuthProvider()
    project=...,    # GCP project ID
    location=...,   # GCP region
    vertexai=...,   # Use Vertex AI (default: True) vs. AI Studio
)
```
