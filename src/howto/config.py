"""Central configuration: API keys, teacher/judge model ids, student candidates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = REPO_ROOT / "runs"
REPORTS_DIR = REPO_ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, RUNS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# .env in this repo uses lowercase keys; normalise to the SDK-standard names.
load_dotenv(REPO_ROOT / ".env")
for _lower, _upper in (
    ("openai_api_key", "OPENAI_API_KEY"),
    ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    # Gated repos (Gemma 3) need this exported, not just present in .env.
    ("hf_token", "HF_TOKEN"),
    ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
):
    if os.environ.get(_lower) and not os.environ.get(_upper):
        os.environ[_upper] = os.environ[_lower].strip().strip('"')


@dataclass(frozen=True)
class ProviderSpec:
    """A closed-weight model used as proposer (teacher) and/or blind judge."""

    key: str
    provider: str
    model: str
    # Prompt-level style hints give proposal diversity when temperature is pinned.
    style_hints: tuple[str, ...] = (
        "Prefer the most direct single-utility solution.",
        "Prefer a portable POSIX-compatible solution.",
    )


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(key="openai", provider="openai", model="gpt-5.6-luna"),
    "anthropic": ProviderSpec(
        key="anthropic", provider="anthropic", model="claude-haiku-4-5-20251001"
    ),
    # Vertex AI, inference only (generateContent). gemini-3-flash-lite does not
    # exist as a publisher model on these projects; 3.5-flash-lite does.
    "gemini": ProviderSpec(
        key="gemini", provider="gemini", model="gemini-3.5-flash-lite"
    ),
}

# Vertex AI routing for the Gemini provider. Overridable via env.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "ml-experiments-d2f6212005a0")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

# google-auth warns on every client construction when user ADC carries no quota
# project. Naming one here billing-attributes the calls and removes the warning
# at its source -- this only annotates the local credential, it changes nothing
# in the cloud. Must be set before google.auth is first imported.
os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", VERTEX_PROJECT)


@dataclass(frozen=True)
class StudentSpec:
    """A small distillation candidate trained locally on the 5090."""

    key: str
    hf_id: str
    lora_r: int = 32
    lora_alpha: int = 64
    learning_rate: float = 2e-4
    # Roughly how much VRAM/time this costs; used to order the tournament.
    tier: str = "nano"
    # "autoregressive" decodes one token at a time; "block_diffusion" denoises a
    # whole block in parallel, which is the interesting latency story for a
    # one-line-command task. The two need different generate() paths.
    decoder: str = "autoregressive"
    # Gated repos need `huggingface-cli login` plus license acceptance on the hub.
    gated: bool = False
    # Repos shipping custom modeling code require trust_remote_code=True.
    remote_code: bool = False
    notes: str = ""
    # PEFT matches module-name suffixes, or a regex when given a single string.
    # Gemma 4 wraps each projection in Gemma4ClippableLinear, so the adapter has
    # to attach to the inner .linear rather than the wrapper.
    lora_targets: tuple[str, ...] | str = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


# Tournament candidates. Every entry is a stock open-weight base/instruct model
# that we fine-tune ourselves -- no already-fine-tuned howto checkpoint is ever
# used as a starting point. Those published checkpoints are eval-only baselines.
#
# The PoC defaults to the two cheapest so a full generate -> train -> evaluate
# loop fits in minutes; larger ones are opt-in via CANDIDATES=... .
STUDENTS: dict[str, StudentSpec] = {
    # nano track (<1B)
    "qwen3-0.6b": StudentSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", tier="nano"),
    "qwen25-coder-0.5b": StudentSpec(
        "qwen25-coder-0.5b", "Qwen/Qwen2.5-Coder-0.5B-Instruct", tier="nano"
    ),
    # core track (1.7-2B): the plan's primary target band
    "qwen3-1.7b": StudentSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", tier="core"),
    # Qwen3.5 nano + core, base and instruct. Base variants ship a chat template
    # so the same one-line-command contract applies to all four.
    "qwen35-0.8b-base": StudentSpec(
        "qwen35-0.8b-base", "Qwen/Qwen3.5-0.8B-Base", tier="nano"
    ),
    "qwen35-0.8b": StudentSpec("qwen35-0.8b", "Qwen/Qwen3.5-0.8B", tier="nano"),
    "qwen35-2b-base": StudentSpec(
        "qwen35-2b-base", "Qwen/Qwen3.5-2B-Base", tier="core"
    ),
    "qwen35-2b": StudentSpec("qwen35-2b", "Qwen/Qwen3.5-2B", tier="core"),
    # family-diverse control: tests whether gains come from the data/method
    # rather than from Qwen-family knowledge. E2B is ~2B effective params.
    # Gemma 4 is a multimodal checkpoint: a bare "q_proj" suffix also matches the
    # vision tower (whose Gemma4ClippableLinear PEFT cannot wrap, and which never
    # runs in a text-only forward pass -> zero gradients). Scope to the language
    # model, where the projections are plain nn.Linear.
    "gemma4-e2b": StudentSpec(
        "gemma4-e2b", "google/gemma-4-E2B-it", tier="core", learning_rate=1e-4,
        lora_targets=(
            r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj"
            r"|gate_proj|up_proj|down_proj)$"
        ),
    ),
    # same-family size control (this is stock Qwen2.5-Coder, not a howto tune)
    "qwen25-coder-1.5b": StudentSpec(
        "qwen25-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct", tier="core"
    ),
    # Gemma 3 is license-gated on the hub: accept the terms once, then
    # `huggingface-cli login`, or these will 401 on download.
    "gemma3-1b": StudentSpec(
        "gemma3-1b", "google/gemma-3-1b-it", tier="nano", gated=True,
        learning_rate=1e-4,
    ),
    "gemma3-4b": StudentSpec(
        "gemma3-4b", "google/gemma-3-4b-it", tier="ceiling", gated=True,
        learning_rate=1e-4,
    ),
    # --- parallel-decoding track -------------------------------------------
    # Google ships exactly one block-diffusion Gemma (26B-A4B, 51.6 GB bf16),
    # and no small one. The practical ~1B parallel decoder is an
    # autoregressive-to-diffusion adapted Qwen: same size class, already
    # converted, and our verified data trains it the same way.
    # These need a block-diffusion generate() path -- see DIFFUSION_TRACK below.
    "qwen3-0.6b-bd3lm": StudentSpec(
        "qwen3-0.6b-bd3lm", "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
        tier="nano", decoder="block_diffusion", remote_code=True,
        notes="AR->diffusion adapted Qwen3-0.6B, block-diffusion decoding",
    ),
    "qwen25-coder-0.5b-bd3lm": StudentSpec(
        "qwen25-coder-0.5b-bd3lm",
        "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1",
        tier="nano", decoder="block_diffusion", remote_code=True,
        notes="AR->diffusion adapted Qwen2.5-Coder-0.5B",
    ),
}

# The 26B block-diffusion Gemma is a teacher, not a student: 51.6 GB in bf16
# exceeds the 5090's 32 GB, so it must be loaded 4-bit (~13 GB) or via one of
# the NVFP4/AWQ-INT4 community builds.
DIFFUSION_TEACHER = "google/diffusiongemma-26B-A4B-it"

DEFAULT_CANDIDATES = ("qwen3-0.6b", "qwen25-coder-0.5b")

# Deployment/eval contract: the student emits exactly one line of shell, no prose.
SYSTEM_PROMPT = (
    "You translate a natural-language request into a single POSIX/bash command line.\n"
    "Rules:\n"
    "- Output ONLY the command. No prose, no markdown fences, no explanation.\n"
    "- The command runs with the working directory set to the task workspace.\n"
    "- Do not use sudo, network access, or destructive operations unless explicitly asked.\n"
)


@dataclass
class SandboxLimits:
    timeout_s: float = 10.0
    max_output_bytes: int = 64_000
    max_file_size_mb: int = 32
    max_processes: int = 256
    address_space_mb: int = 2048
    replay_seeds: int = 3
    env: dict[str, str] = field(
        default_factory=lambda: {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "HOME": "/work",
            "SHELL": "/bin/bash",
            "TERM": "dumb",
        }
    )


LIMITS = SandboxLimits()
