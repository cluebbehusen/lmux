# lmux-gcp-vertex (deprecated)

**This package has been renamed to [lmux-google](https://pypi.org/project/lmux-google/).**

The rename reflects what the package actually is: a provider for Google's models via the [google-genai](https://pypi.org/project/google-genai/) SDK, which serves Gemini through either Vertex AI or the Gemini Developer API. Vertex AI has no unified inference API across publishers, so partner models hosted on Vertex (Anthropic Claude, Mistral, Llama/DeepSeek/Grok MaaS, ...) were never servable through this package — "Vertex" was the wrong name for it.

This final release is a compatibility shim: it depends on `lmux-google` and re-exports everything under the old names, emitting a `DeprecationWarning` on import. It will not receive further updates.

## Migration

| Old (`lmux_gcp_vertex`)               | New (`lmux_google`)                |
| ------------------------------------- | ---------------------------------- |
| `GCPVertexProvider`                   | `GoogleProvider`                   |
| `GCPVertexParams`                     | `GoogleParams`                     |
| `GCPVertexADCAuthProvider`            | `GoogleADCAuthProvider`            |
| `GCPVertexAPIKeyAuthProvider`         | `GoogleAPIKeyAuthProvider`         |
| `GCPVertexServiceAccountAuthProvider` | `GoogleServiceAccountAuthProvider` |
| `calculate_gcp_vertex_cost`           | `calculate_google_cost`            |

All other exports (`GoogleSearchConfig`, `SafetySetting`, `preload`, ...) keep their names.

## Behavior changes vs. 0.6.x

- Responses now report `provider="google"` instead of `provider="gcp-vertex"`.
- Pricing entries for partner models (Claude, Mistral, Llama, DeepSeek, Grok, Qwen, GLM, and other Vertex MaaS models) were removed — those models were never callable through this provider, so the entries were unreachable dead data. `calculate_gcp_vertex_cost` now returns `None` for them.
