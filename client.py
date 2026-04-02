"""
Example client for the Qwen Modal endpoint.

Uses the OpenAI SDK since vLLM exposes an OpenAI-compatible API.

Usage:
    # Set your endpoint URL (printed by `modal deploy serve.py`)
    export QWEN_BASE_URL="https://your-app--qwen-serving-serve.modal.run"

    python client.py
    python client.py "What is the capital of France?"
"""

import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("QWEN_BASE_URL")
if not BASE_URL:
    print("Set QWEN_BASE_URL to your Modal endpoint URL")
    print('  export QWEN_BASE_URL="https://your-app--qwen-serving-serve.modal.run"')
    sys.exit(1)

# Modal endpoints don't require an API key, but the OpenAI SDK needs one set
client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain mixture-of-experts in 3 sentences."

print(f"Model: {MODEL}")
print(f"Prompt: {prompt}\n")

stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    stream=True,
    max_tokens=2048,
)

for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="", flush=True)
print()
