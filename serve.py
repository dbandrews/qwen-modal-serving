"""
Qwen 3.5 LLM serving on Modal with vLLM.

Deploys a Qwen 3.5 model behind an OpenAI-compatible API endpoint.
Use as a shared LLM endpoint across multiple projects.

Deploy:   modal deploy serve.py
Dev run:  modal serve serve.py
Test:     modal run serve.py
"""

import json
from typing import Any

import aiohttp
import modal

# --- Model Configuration ---
# Qwen3.5-122B-A10B-FP8: MoE model, 122B total / 10B active params.
# Strong quality with efficient sparse activation. Requires 2x H100 (80GB) with FP8.
MODEL_NAME = "Qwen/Qwen3.5-122B-A10B-FP8"
MODEL_REVISION = None  # pin a commit hash for reproducibility, or None for latest
N_GPU = 2
GPU_TYPE = "H100"
MAX_MODEL_LEN = 32768  # 32K context; increase if you need longer (up to 262144 native)

# --- Alternative configurations (uncomment one block to switch) ---
#
# Qwen3.5-397B-A17B-FP8: Flagship MoE. 397B total / 17B active. Best quality.
# MODEL_NAME = "Qwen/Qwen3.5-397B-A17B-FP8"
# N_GPU = 4
# GPU_TYPE = "H100"
# MAX_MODEL_LEN = 32768
#
# Qwen3.5-35B-A3B-FP8: MoE, 35B total / 3B active. Best throughput/cost ratio.
# MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
# N_GPU = 1
# GPU_TYPE = "H100"
# MAX_MODEL_LEN = 32768
#
# Qwen3.5-27B-FP8: Dense 27B model. Strong quality, fits on H100 or L40S with FP8.
# MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
# N_GPU = 1
# GPU_TYPE = "H100"  # or "L40S" ($1.95/hr) -- tight on VRAM with long context
# MAX_MODEL_LEN = 32768
#
# Qwen3.5-9B: Smallest practical model. Fits on L40S in bf16. Budget option.
# MODEL_NAME = "Qwen/Qwen3.5-9B"
# N_GPU = 1
# GPU_TYPE = "L40S"  # $1.95/hr, or "A10" ($1.10/hr) for 24GB -- tight
# MAX_MODEL_LEN = 32768

# --- Modal setup ---
MINUTES = 60
VLLM_PORT = 8000

# Qwen3.5 uses Gated DeltaNet architecture, which requires nightly vLLM builds
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm",
        extra_options="--torch-backend=cu126 --extra-index-url https://wheels.vllm.ai/nightly",
    )
    .uv_pip_install("huggingface-hub[hf_xet]>=0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen-serving")

# CUDA graphs reduce per-token latency significantly once warm.
# Cold start is longer (~5-8 min extra for graph capture), but generation is faster.
FAST_BOOT = False


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        MODEL_NAME,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(N_GPU),
        "--language-model-only",  # text-only mode, saves VRAM; remove for multimodal
        "--reasoning-parser",
        "qwen3",  # enables <think> tag parsing for thinking mode
        "--kv-cache-dtype",
        "fp8",  # halves KV cache memory vs bf16
        "--gpu-memory-utilization",
        "0.95",  # use more VRAM (default 0.9)
        "--enable-prefix-caching",  # reuse KV cache for repeated prefixes (e.g. system prompt)
        "--gdn-prefill-backend",
        "triton",  # avoids ~5 min FlashInfer JIT compile on cold start
        "--uvicorn-log-level=info",
    ]

    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    if FAST_BOOT:
        cmd.append("--enforce-eager")

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# --- Test entrypoint ---


@app.local_entrypoint()
async def test(content: str = None, timeout: int = 10 * MINUTES):
    """Quick smoke test: hit the deployed endpoint with a test prompt."""
    import asyncio

    url = await serve.get_web_url.aio()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content or "Explain what a mixture-of-experts model is in 3 sentences.",
        },
    ]

    req_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(base_url=url, timeout=req_timeout) as session:
        # Retry health check — server may still be loading the model after cold start
        print(f"Waiting for server at {url} ...")
        for attempt in range(1, 61):
            try:
                async with session.get("/health") as resp:
                    if resp.status == 200:
                        print(f"Health check passed (attempt {attempt})")
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if attempt == 60:
                raise RuntimeError("Health check failed after 60 attempts (~5 min)")
            await asyncio.sleep(5)

        print(f"Sending request to {url}")
        await _stream_chat(session, MODEL_NAME, messages)


async def _stream_chat(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            chunk = json.loads(line)
            token = chunk["choices"][0]["delta"].get("content", "")
            print(token, end="", flush=True)
    print()
