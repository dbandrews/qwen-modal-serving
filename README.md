# Qwen 3.5 on Modal (vLLM)

Serves Qwen 3.5 models via vLLM on Modal, exposing an OpenAI-compatible API endpoint.

## Setup

```bash
pip install modal
modal setup  # authenticate
```

## Deploy

```bash
# Development (hot-reload)
modal serve serve.py

# Production
modal deploy serve.py
```

The deploy command prints your endpoint URL.

## Use

The endpoint is OpenAI-compatible. Use any OpenAI SDK client:

```bash
export QWEN_BASE_URL="https://your-app--qwen-serving-serve.modal.run"
pip install openai
python client.py "Your prompt here"
```

Or from any project with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Interactive API docs at `{BASE_URL}/docs`.

## Model options

| Model | Type | VRAM (approx) | GPU | Cost/hr | Notes |
|-------|------|---------------|-----|---------|-------|
| `Qwen/Qwen3.5-35B-A3B-FP8` | MoE | ~36GB | H100 | ~$3.95 | **Default.** 3B active params = fast inference |
| `Qwen/Qwen3.5-27B-FP8` | Dense | ~28GB | H100 or L40S | $1.95-$3.95 | Strong quality, cheaper on L40S |
| `Qwen/Qwen3.5-9B` | Dense | ~20GB (bf16) | L40S | ~$1.95 | Budget option |

Switch models by editing the config block at the top of `serve.py`.

## Notes

- Qwen 3.5 uses Gated DeltaNet architecture, requiring **nightly vLLM** (not stable releases)
- Default config uses `--language-model-only` (text-only). Remove that flag in `serve.py` for vision/multimodal
- Scales to zero when idle (15 min window). First request after idle triggers cold start (~2-5 min)
- Supports thinking mode (`<think>` tags) via `--reasoning-parser qwen3`
