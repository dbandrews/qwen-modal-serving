"""
Qwen LLM serving on Modal with SGLang and GPU memory snapshots.

Deploys a Qwen model behind an OpenAI-compatible API endpoint.
Uses GPU snapshotting for ~10x faster cold starts via SGLang's
memory offload (sleep/wake) and Modal's snapshot lifecycle hooks.

Deploy:   modal deploy serve.py
Test:     modal run serve.py
"""

import subprocess
import time
from typing import Any

import aiohttp
import modal

MINUTES = 60
PORT = 8000

# --- Model Configuration ---
# Qwen3.5-35B-A3B-FP8: MoE, 35B total / 3B active, FP8 quantized (~35GB).
# Uses Gated DeltaNet architecture, supported in SGLang v0.5.9+.
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION = None  # pin a commit hash for reproducibility, or None for latest
N_GPUS = 1
GPU = f"H100:{N_GPUS}"
MAX_MODEL_LEN = 32768

# --- Autoscaling ---
TARGET_INPUTS = 10
MAX_INPUTS = 32

# --- SGLang image ---
sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.9-cu129-amd64-runtime"
    )
    .entrypoint([])
    .uv_pip_install("huggingface-hub[hf_xet]>=0.36.0")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",  # improve torch.compile snapshot compat
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    })
)

# --- Volumes ---
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# DeepGEMM JIT compilation cache (H100 only)
dg_cache_vol = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"


def compile_deep_gemm():
    """Pre-compile DeepGEMM kernels during image build to avoid JIT at runtime."""
    subprocess.run(
        f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --tp {N_GPUS}",
        shell=True,
    )


sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: dg_cache_vol, HF_CACHE_PATH: hf_cache_vol},
    gpu=GPU,
    timeout=30 * MINUTES,
)

app = modal.App("qwen-serving")

# --- Snapshot helpers (run inside container) ---

with sglang_image.imports():
    import requests as _requests


def warmup():
    """Send a few requests to fully warm up the server before snapshotting."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        _requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=60,
        ).raise_for_status()


def sleep_server():
    """Move non-essential data out of GPU memory before snapshotting."""
    _requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_server():
    """Restore GPU memory after snapshot restore."""
    _requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES):
    """Block until SGLang server is healthy or raise on failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if (rc := process.poll()) is not None:
                raise subprocess.CalledProcessError(rc, cmd=process.args)
            _requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            _requests.exceptions.ConnectionError,
            _requests.exceptions.HTTPError,
        ):
            time.sleep(1)
    raise TimeoutError(f"SGLang server not ready within {timeout}s")


# --- Inference server ---


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: hf_cache_vol, DG_CACHE_PATH: dg_cache_vol},
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class Inference:
    @modal.enter(snap=True)
    def startup(self):
        """Start SGLang, wait for health, warm up, then sleep for snapshot."""
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tp",
            str(N_GPUS),
            "--context-length",
            str(MAX_MODEL_LEN),
            "--cuda-graph-max-bs",
            str(MAX_INPUTS),
            "--max-running-requests",
            str(MAX_INPUTS),
            "--tool-call-parser",
            "qwen",  # Qwen tool calling format
            "--enable-metrics",
            "--enable-memory-saver",  # enable offload for snapshotting
            "--enable-weights-cpu-backup",  # enable offload for snapshotting
        ]
        if MODEL_REVISION:
            cmd += ["--revision", MODEL_REVISION]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        sleep_server()

    @modal.enter(snap=False)
    def wake(self):
        """Restore GPU memory on snapshot resume."""
        wake_server()

    @modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.process.terminate()


# --- Test entrypoint ---


@app.local_entrypoint()
async def test(content: str = None, timeout: int = 10 * MINUTES):
    """Quick smoke test: hit the deployed endpoint with a test prompt."""
    import asyncio
    import json

    url = Inference().serve.get_web_url()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content
            or "Explain what a mixture-of-experts model is in 3 sentences.",
        },
    ]

    req_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(base_url=url, timeout=req_timeout) as session:
        print(f"Waiting for server at {url} ...")
        for attempt in range(1, 121):
            try:
                async with session.get("/health") as resp:
                    if resp.status == 200:
                        print(f"Health check passed (attempt {attempt})")
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if attempt == 120:
                raise RuntimeError(
                    "Health check failed after 120 attempts (~10 min)"
                )
            await asyncio.sleep(5)

        print(f"Sending request to {url}")
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
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
