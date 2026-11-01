"""Reproducible, download-gated GGUF export for HowTo deployment tiers.

The exporter is deliberately offline by default.  It will only materialise Git
LFS objects, contact Hugging Face, or clone llama.cpp when the operator passes
``--allow-downloads``.  Every input that affects the result is pinned or hashed
and recorded next to the quantized artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .prompts import (
    PRODUCTION_PROMPT_POLICIES,
    PRODUCTION_RUNTIME_TARGETS,
    TRAINING_PROMPT_POLICY_ID,
    TRAINING_SYSTEM_PROMPT_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
QWEN35_LORA_PR_REVISION = "f839835a3401f0bf000d362dc10ba6b1c50d3a3f"
# Exact llama.cpp commit vendored by HowTo's embedded llama-cpp-rs runtime.
RUNTIME_LLAMA_CPP_REVISION = "5f55650a78f92aff4d48d671423e888fac0469ff"
GENERATION_POLICY_ID = "howto.embedded.command-generation.deterministic.v1"
QUALIFICATION_TASK_CORPUS_SHA256 = (
    "2457970a5bc708ffe642cf62ceb430e80c0493c78ab40d7caa4f71acd44fe204"
)
QUALIFICATION_SPLIT_POLICY_ID = "howto.group-kfold.fold-0.v1"
QUALIFICATION_SPLIT_POLICY_SHA256 = (
    "d23e9d22519f8d0aee184be49f179010e6db78d37889486f1a713e114367a684"
)
QUALIFICATION_SPLIT_POLICY: dict[str, Any] = {
    "algorithm": "sklearn.model_selection.GroupKFold",
    "fold": 0,
    "group_key": "howto.splits.group_key.v1",
    "n_splits": 5,
    "source_task_corpus_sha256": QUALIFICATION_TASK_CORPUS_SHA256,
}
EXPERIMENTAL_LLAMA_CPP_REVISIONS = {
    QWEN35_LORA_PR_REVISION: {
        "source_status": "unmerged_experimental",
        "upstream_pr": "https://github.com/ggml-org/llama.cpp/pull/24627",
        "fetch_ref": "pull/24627/head",
    }
}
QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q6_K")
DUAL_STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")
PROMPT_CANARY_REQUEST = "__HOWTO_USER_REQUEST__"
BASE_METADATA_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_templates/*.jinja",
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


class ExportError(RuntimeError):
    """A safe, operator-actionable export refusal."""


@dataclass(frozen=True)
class DeploymentSpec:
    id: str
    tier: str
    display_name: str
    base_model_id: str
    adapter_relative_path: Path
    parameters_billions: float
    measured_executable_pass_at_1: float
    measured_mean_latency_ms: float
    measured_peak_bf16_gb: float
    recommended_minimum_ram_bytes: int
    prompt_format_id: str
    assistant_generation_prefix: str
    expected_stop_ids: tuple[int, int]
    expected_rendered_prefix_sha256: str
    expected_generation_policy_sha256: str
    recommended_quantization: str = "Q4_K_M"

    @property
    def artifact_stem(self) -> str:
        return f"howto-{self.id}"


# Only measured Pareto points are offered to setup. Base variants and models
# dominated on both quality and footprint stay in the experiment repository.
DEPLOYMENT_SPECS: dict[str, DeploymentSpec] = {
    "compact": DeploymentSpec(
        id="qwen25-coder-0.5b",
        tier="compact",
        display_name="Compact — Qwen2.5 Coder 0.5B",
        base_model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        adapter_relative_path=Path("runs/qwen25-coder-0.5b-fold0"),
        parameters_billions=0.5,
        measured_executable_pass_at_1=0.6375,
        measured_mean_latency_ms=86.4,
        measured_peak_bf16_gb=1.14,
        recommended_minimum_ram_bytes=1_610_612_736,
        prompt_format_id="qwen_chatml_no_thinking",
        assistant_generation_prefix="<|im_start|>assistant\n",
        expected_stop_ids=(151643, 151645),
        expected_rendered_prefix_sha256=(
            "3c8662b24dbbc7cbf131e39538fa52fca4f848250a36088c64b62008dc655487"
        ),
        expected_generation_policy_sha256=(
            "bf5cc2fc03580a33d0a71f19153a0f52ed35715adcddb1277e2326b77afbf35e"
        ),
    ),
    "balanced": DeploymentSpec(
        id="qwen35-0.8b",
        tier="balanced",
        display_name="Balanced — Qwen3.5 0.8B",
        base_model_id="Qwen/Qwen3.5-0.8B",
        adapter_relative_path=Path("runs/qwen35-0.8b-fold0"),
        parameters_billions=0.8,
        measured_executable_pass_at_1=0.6775,
        measured_mean_latency_ms=180.0,
        measured_peak_bf16_gb=1.62,
        recommended_minimum_ram_bytes=2_147_483_648,
        prompt_format_id="qwen35_no_thinking",
        assistant_generation_prefix=(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        expected_stop_ids=(248044, 248046),
        expected_rendered_prefix_sha256=(
            "85faff74b0cd134fac958e43cea0b514bb50b3170e9c95d2d0c355d578c77577"
        ),
        expected_generation_policy_sha256=(
            "65b4d051714ee98d3c1e405af9e84b77230637e188fbe66695135ff0673a4660"
        ),
    ),
    "accurate": DeploymentSpec(
        id="qwen35-2b",
        tier="accurate",
        display_name="Accurate — Qwen3.5 2B",
        base_model_id="Qwen/Qwen3.5-2B",
        adapter_relative_path=Path("runs/qwen35-2b-fold0"),
        parameters_billions=2.0,
        measured_executable_pass_at_1=0.745,
        measured_mean_latency_ms=181.8,
        measured_peak_bf16_gb=3.96,
        recommended_minimum_ram_bytes=4_294_967_296,
        prompt_format_id="qwen35_no_thinking",
        assistant_generation_prefix=(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        expected_stop_ids=(248044, 248046),
        expected_rendered_prefix_sha256=(
            "85faff74b0cd134fac958e43cea0b514bb50b3170e9c95d2d0c355d578c77577"
        ),
        expected_generation_policy_sha256=(
            "65b4d051714ee98d3c1e405af9e84b77230637e188fbe66695135ff0673a4660"
        ),
    ),
}


@dataclass(frozen=True)
class LfsPointer:
    sha256: str
    size: int


@dataclass(frozen=True)
class AdapterProvenance:
    revision: str
    remote: str | None
    model_sha256: str
    model_size: int
    tokenizer_sha256: str
    tokenizer_size: int
    template_sha256: str
    metadata_sha256: dict[str, str]
    training_base_revision: str | None = None


@dataclass(frozen=True)
class LlamaToolchain:
    revision: str
    converter: Path
    lora_converter: Path
    quantizer: Path | None
    converter_sha256: str
    lora_converter_sha256: str
    quantizer_sha256: str | None
    source_status: str = "unqualified"
    upstream_pr: str | None = None
    production_approved: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_lfs_pointer(value: bytes) -> LfsPointer | None:
    """Return pointer metadata, or ``None`` when *value* is real file data."""
    if not value.startswith(LFS_HEADER):
        return None
    text = value.decode("ascii", errors="strict")
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if oid is None or size is None:
        raise ExportError("malformed Git LFS pointer")
    return LfsPointer(sha256=oid.group(1), size=int(size.group(1)))


def pointer_at_path(path: Path) -> LfsPointer | None:
    if not path.is_file():
        return None
    with path.open("rb") as source:
        return parse_lfs_pointer(source.read(512))


def immutable_revision(value: str, label: str) -> str:
    revision = value.strip().lower()
    if not REVISION_RE.fullmatch(revision):
        raise ExportError(
            f"{label} must be a full immutable 40-hex (or 64-hex) commit, "
            "not a branch, tag, or short SHA"
        )
    return revision


def llama_cpp_source_provenance(revision: str) -> dict[str, str | bool | None]:
    metadata = EXPERIMENTAL_LLAMA_CPP_REVISIONS.get(revision, {})
    return {
        "source_status": str(metadata.get("source_status", "unqualified")),
        "upstream_pr": (
            str(metadata["upstream_pr"]) if metadata.get("upstream_pr") else None
        ),
        "production_approved": False,
    }


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(argv), flush=True)
    try:
        subprocess.run(list(argv), cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise ExportError(f"required executable not found: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ExportError(
            f"command failed with exit code {exc.returncode}: {' '.join(argv)}"
        ) from exc


def _capture(argv: Sequence[str], *, cwd: Path | None = None) -> bytes:
    try:
        result = subprocess.run(
            list(argv), cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise ExportError(f"required executable not found: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise ExportError(f"command failed: {' '.join(argv)}: {detail}") from exc
    return result.stdout


def _git_blob(repo_root: Path, revision: str, relative_path: Path) -> bytes:
    return _capture(
        ["git", "show", f"{revision}:{relative_path.as_posix()}"], cwd=repo_root
    )


def _materialize_lfs_files(
    repo_root: Path,
    relative_paths: Sequence[Path],
    *,
    allow_downloads: bool,
    runner: Callable[..., None] = _run,
) -> None:
    missing_or_pointer = [
        path
        for path in relative_paths
        if not (repo_root / path).is_file() or pointer_at_path(repo_root / path) is not None
    ]
    if not missing_or_pointer:
        return

    display = ", ".join(path.as_posix() for path in missing_or_pointer)
    if not allow_downloads:
        raise ExportError(
            "adapter inputs are absent or still Git LFS pointers: "
            f"{display}. Re-run with --allow-downloads to permit a targeted "
            "Git LFS fetch. No download was attempted."
        )
    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        try:
            _capture(["git", "lfs", "version"], cwd=repo_root)
        except ExportError as exc:
            raise ExportError(
                "--allow-downloads was set, but git-lfs is not installed; install "
                "Git LFS and retry"
            ) from exc
    include = ",".join(path.as_posix() for path in missing_or_pointer)
    runner(
        ["git", "lfs", "pull", f"--include={include}", "--exclude="],
        cwd=repo_root,
    )


def _verify_lfs_object(path: Path, expected: LfsPointer) -> None:
    if not path.is_file():
        raise ExportError(f"required adapter file is missing: {path}")
    if pointer_at_path(path) is not None:
        raise ExportError(f"Git LFS did not materialise adapter file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected.size:
        raise ExportError(
            f"adapter file size mismatch for {path}: expected {expected.size}, "
            f"got {actual_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected.sha256:
        raise ExportError(
            f"adapter file checksum mismatch for {path}: expected "
            f"{expected.sha256}, got {actual_sha}"
        )


def inspect_adapter(
    repo_root: Path,
    adapter_dir: Path,
    spec: DeploymentSpec,
    *,
    expected_revision: str | None,
    allow_downloads: bool,
) -> AdapterProvenance:
    """Validate that every merge input matches one immutable repository commit."""
    repo_root = repo_root.resolve()
    adapter_dir = adapter_dir.resolve()
    try:
        adapter_relative = adapter_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ExportError("adapter directory must be inside the modelling repository") from exc

    revision = _capture(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    revision = immutable_revision(revision, "adapter revision")
    if expected_revision is not None:
        wanted = immutable_revision(expected_revision, "expected adapter revision")
        if revision != wanted:
            raise ExportError(
                f"adapter checkout is {revision}, but --adapter-revision requires {wanted}"
            )

    lfs_names = ("adapter_model.safetensors", "tokenizer.json")
    lfs_relative = [adapter_relative / name for name in lfs_names]
    expected_pointers: dict[str, LfsPointer] = {}
    for name, relative in zip(lfs_names, lfs_relative):
        committed = parse_lfs_pointer(_git_blob(repo_root, revision, relative))
        if committed is None:
            raise ExportError(
                f"expected {relative} to be an immutable Git LFS object at {revision}"
            )
        expected_pointers[name] = committed

    _materialize_lfs_files(
        repo_root, lfs_relative, allow_downloads=allow_downloads
    )
    for name in lfs_names:
        _verify_lfs_object(adapter_dir / name, expected_pointers[name])

    metadata_names = (
        "adapter_config.json",
        "run_meta.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    metadata_sha: dict[str, str] = {}
    for name in metadata_names:
        path = adapter_dir / name
        if not path.is_file():
            raise ExportError(f"required adapter metadata is missing: {path}")
        local = path.read_bytes()
        committed = _git_blob(repo_root, revision, adapter_relative / name)
        if local != committed:
            raise ExportError(
                f"adapter metadata differs from immutable revision {revision}: {path}"
            )
        metadata_sha[name] = sha256_bytes(local)

    try:
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        run_meta = json.loads((adapter_dir / "run_meta.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid adapter metadata: {exc}") from exc
    configured_base = adapter_config.get("base_model_name_or_path")
    trained_base = run_meta.get("hf_id")
    if configured_base != spec.base_model_id or trained_base != spec.base_model_id:
        raise ExportError(
            "selected adapter does not target the required base model: "
            f"adapter_config={configured_base!r}, run_meta={trained_base!r}, "
            f"expected={spec.base_model_id!r}"
        )
    recorded_base_revision = run_meta.get("base_revision") or adapter_config.get(
        "revision"
    )
    if recorded_base_revision is not None:
        recorded_base_revision = immutable_revision(
            str(recorded_base_revision), "recorded training base revision"
        )

    try:
        remote = _capture(["git", "remote", "get-url", "origin"], cwd=repo_root)
        remote_text: str | None = remote.decode().strip() or None
    except ExportError:
        remote_text = None

    return AdapterProvenance(
        revision=revision,
        remote=remote_text,
        model_sha256=expected_pointers["adapter_model.safetensors"].sha256,
        model_size=expected_pointers["adapter_model.safetensors"].size,
        tokenizer_sha256=expected_pointers["tokenizer.json"].sha256,
        tokenizer_size=expected_pointers["tokenizer.json"].size,
        template_sha256=metadata_sha["chat_template.jinja"],
        metadata_sha256=metadata_sha,
        training_base_revision=recorded_base_revision,
    )


def verify_measured_metrics(
    repo_root: Path, revision: str, spec: DeploymentSpec
) -> dict[str, Any]:
    """Bind catalog claims to the immutable reports that produced them."""
    report_paths = (Path("reports/evaluation.json"), Path("reports/latency.json"))
    payloads: dict[Path, bytes] = {}
    for relative in report_paths:
        local = (repo_root / relative).read_bytes()
        committed = _git_blob(repo_root, revision, relative)
        if local != committed:
            raise ExportError(
                f"deployment metrics differ from immutable revision {revision}: {relative}"
            )
        payloads[relative] = local

    try:
        evaluations = json.loads(payloads[report_paths[0]])
        latencies = json.loads(payloads[report_paths[1]])
    except json.JSONDecodeError as exc:
        raise ExportError(f"invalid deployment metrics report: {exc}") from exc
    evaluation = [row for row in evaluations if row.get("model") == f"{spec.id}-sft"]
    latency = [
        row
        for row in latencies
        if row.get("model") == spec.id
        and row.get("tuned") is True
        and row.get("variant") == "sft"
    ]
    if len(evaluation) != 1 or len(latency) != 1:
        raise ExportError(f"expected one immutable evaluation and latency row for {spec.id}")
    expected = {
        "pass@1": spec.measured_executable_pass_at_1,
        "mean_ms": spec.measured_mean_latency_ms,
        "peak_gpu_gb": spec.measured_peak_bf16_gb,
    }
    actual = {
        "pass@1": evaluation[0].get("pass@1"),
        "mean_ms": latency[0].get("mean_ms"),
        "peak_gpu_gb": latency[0].get("peak_gpu_gb"),
    }
    if actual != expected or evaluation[0].get("n") != 400:
        raise ExportError(
            f"catalog metrics for {spec.id} do not match immutable reports: "
            f"expected {expected} at n=400, got {actual} at n={evaluation[0].get('n')}"
        )
    return {
        "repository_revision": revision,
        "evaluation": {
            "path": report_paths[0].as_posix(),
            "sha256": sha256_bytes(payloads[report_paths[0]]),
        },
        "latency": {
            "path": report_paths[1].as_posix(),
            "sha256": sha256_bytes(payloads[report_paths[1]]),
        },
    }


def resolve_base_snapshot(
    model_id: str,
    revision: str,
    *,
    cache_dir: Path | None,
    allow_downloads: bool,
    metadata_only: bool = False,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Resolve one exact Hub snapshot, with network disabled unless opted in.

    ``metadata_only`` asks the Hub for the small configuration/tokenizer subset
    used by llama.cpp's standalone LoRA converter. It never includes a model
    weight filename, even when downloads have been explicitly enabled.
    """
    revision = immutable_revision(revision, "base model revision")
    if not allow_downloads:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ExportError(
                "huggingface-hub is required; run `uv sync --extra export`"
            ) from exc
        downloader = snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "revision": revision,
        "local_files_only": not allow_downloads,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if metadata_only:
        kwargs["allow_patterns"] = list(BASE_METADATA_PATTERNS)
    try:
        snapshot = Path(downloader(**kwargs)).resolve()
    except Exception as exc:
        mode = "download-enabled" if allow_downloads else "local-only"
        raise ExportError(
            f"could not resolve {model_id}@{revision} in {mode} mode: {exc}"
        ) from exc

    if snapshot.name.lower() != revision:
        raise ExportError(
            f"Hub returned snapshot {snapshot}, which is not the requested immutable "
            f"revision {revision}"
        )
    if not (snapshot / "config.json").is_file():
        raise ExportError(f"base snapshot is incomplete (config.json missing): {snapshot}")
    tokenizer_files = (snapshot / "tokenizer.json", snapshot / "tokenizer.model")
    if metadata_only and not any(path.is_file() for path in tokenizer_files):
        raise ExportError(
            f"base metadata snapshot has no tokenizer payload: {snapshot}"
        )
    if not metadata_only:
        weight_files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
        if not weight_files:
            raise ExportError(
                f"full base snapshot has no model weights: {snapshot}. Use "
                "--export-mode lora for the metadata-only deployment candidate."
            )
    return snapshot


def _clone_llama_cpp(destination: Path, revision: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".llama-cpp-clone-", dir=destination.parent
    ) as temporary:
        checkout = Path(temporary) / "checkout"
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth=1",
                LLAMA_CPP_REPOSITORY,
                str(checkout),
            ]
        )
        metadata = EXPERIMENTAL_LLAMA_CPP_REVISIONS.get(revision, {})
        fetch_ref = str(metadata.get("fetch_ref", revision))
        _run(["git", "fetch", "--depth=1", "origin", fetch_ref], cwd=checkout)
        _run(["git", "checkout", "--detach", revision], cwd=checkout)
        os.replace(checkout, destination)


def _find_quantizer(checkout: Path) -> Path | None:
    candidates = (
        checkout / "build" / "bin" / "llama-quantize",
        checkout / "build" / "bin" / "quantize",
        checkout / "llama-quantize",
        checkout / "quantize",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def prepare_llama_cpp(
    checkout: Path,
    revision: str,
    *,
    allow_downloads: bool,
    build_tools: bool,
    need_quantizer: bool = True,
) -> LlamaToolchain:
    revision = immutable_revision(revision, "llama.cpp revision")
    checkout = checkout.resolve()
    if not checkout.exists():
        if not allow_downloads:
            raise ExportError(
                f"llama.cpp checkout is missing: {checkout}. No clone was attempted; "
                "provide the pinned checkout or pass --allow-downloads."
            )
        _clone_llama_cpp(checkout, revision)
    if not (checkout / ".git").exists():
        raise ExportError(f"not a llama.cpp Git checkout: {checkout}")

    actual = _capture(["git", "rev-parse", "HEAD"], cwd=checkout).decode().strip().lower()
    if actual != revision:
        raise ExportError(
            f"llama.cpp checkout is {actual}, but export is pinned to {revision}; "
            "the exporter will not mutate an existing checkout"
        )
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=checkout, check=False
    )
    cached_diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=checkout,
        check=False,
    )
    if diff.returncode != 0 or cached_diff.returncode != 0:
        raise ExportError("llama.cpp has tracked modifications; use a clean pinned checkout")

    converter = checkout / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise ExportError(f"llama.cpp converter is missing: {converter}")
    lora_converter = checkout / "convert_lora_to_gguf.py"
    if not lora_converter.is_file():
        raise ExportError(f"llama.cpp LoRA converter is missing: {lora_converter}")
    quantizer = _find_quantizer(checkout) if need_quantizer else None
    if need_quantizer and quantizer is None and build_tools:
        build_dir = checkout / "build"
        _run(
            [
                "cmake",
                "-S",
                str(checkout),
                "-B",
                str(build_dir),
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_SERVER=OFF",
                "-DLLAMA_BUILD_EXAMPLES=ON",
                "-DGGML_NATIVE=OFF",
            ]
        )
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "llama-quantize",
                "--parallel",
            ]
        )
        quantizer = _find_quantizer(checkout)
    if need_quantizer and quantizer is None:
        raise ExportError(
            "llama.cpp quantizer is missing. Build target llama-quantize or omit "
            "--no-build-tools."
        )

    source = llama_cpp_source_provenance(revision)
    return LlamaToolchain(
        revision=revision,
        converter=converter,
        lora_converter=lora_converter,
        quantizer=quantizer,
        converter_sha256=sha256_file(converter),
        lora_converter_sha256=sha256_file(lora_converter),
        quantizer_sha256=sha256_file(quantizer) if quantizer is not None else None,
        source_status=str(source["source_status"]),
        upstream_pr=(str(source["upstream_pr"]) if source["upstream_pr"] else None),
        production_approved=bool(source["production_approved"]),
    )


def dual_stop_ids(tokenizer: Any) -> list[dict[str, int | str]]:
    stops: list[dict[str, int | str]] = []
    seen: set[int] = set()
    for token in DUAL_STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            not isinstance(token_id, int)
            or token_id < 0
            or token_id == getattr(tokenizer, "unk_token_id", None)
        ):
            raise ExportError(f"tokenizer does not define required stop token {token!r}")
        if token_id in seen:
            raise ExportError(
                f"required stop tokens resolve to the same id {token_id}: "
                f"{DUAL_STOP_TOKENS}"
            )
        seen.add(token_id)
        stops.append({"token": token, "id": token_id})
    return stops


def validate_stop_ids(
    stops: list[dict[str, int | str]], spec: DeploymentSpec
) -> None:
    actual = tuple(stop["id"] for stop in stops)
    if actual != spec.expected_stop_ids:
        raise ExportError(
            f"{spec.id} stop IDs changed: expected {spec.expected_stop_ids}, "
            f"got {actual}"
        )


def render_deployment_prompt(
    spec: DeploymentSpec, system_prompt: str, user_request: str
) -> str:
    """Render the exact text prefix the embedded runtime must tokenize."""
    request = user_request.strip()
    system = system_prompt if spec.tier == "compact" else system_prompt.strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        f"{spec.assistant_generation_prefix}"
    )


def prompt_contract(spec: DeploymentSpec, system_prompt: str) -> dict[str, Any]:
    rendered = render_deployment_prompt(spec, system_prompt, PROMPT_CANARY_REQUEST)
    return {
        "id": spec.prompt_format_id,
        "system_prompt_policy_id": TRAINING_PROMPT_POLICY_ID,
        "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
        "system_prompt_ends_with_newline": system_prompt.endswith("\n"),
        "request_normalization": "trim",
        "canary_user_request": PROMPT_CANARY_REQUEST,
        "canary_rendered_prefix": rendered,
        "canary_rendered_prefix_sha256": sha256_bytes(rendered.encode("utf-8")),
        "assistant_generation_prefix": spec.assistant_generation_prefix,
        "assistant_generation_prefix_sha256": sha256_bytes(
            spec.assistant_generation_prefix.encode("utf-8")
        ),
        "enable_thinking": False,
        "stop_on_newline": False,
    }


def validate_prompt_rendering(
    tokenizer: Any, spec: DeploymentSpec, system_prompt: str
) -> dict[str, Any]:
    expected = render_deployment_prompt(spec, system_prompt, PROMPT_CANARY_REQUEST)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if spec.prompt_format_id == "qwen35_no_thinking":
        kwargs["enable_thinking"] = False
    actual = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PROMPT_CANARY_REQUEST},
        ],
        **kwargs,
    )
    if actual != expected:
        raise ExportError(
            f"{spec.id} chat rendering no longer matches prompt format "
            f"{spec.prompt_format_id}; refusing to export a runtime contract"
        )
    return prompt_contract(spec, system_prompt)


def validate_prompt_contract(
    contract: dict[str, Any], spec: DeploymentSpec
) -> None:
    system_hash = contract.get("system_prompt_sha256")
    if system_hash != TRAINING_SYSTEM_PROMPT_SHA256:
        raise ExportError(
            f"{spec.id} training system prompt changed: expected "
            f"{TRAINING_SYSTEM_PROMPT_SHA256}, got {system_hash}; the final newline "
            "is part of the trained prompt policy"
        )
    if contract.get("system_prompt_policy_id") != TRAINING_PROMPT_POLICY_ID:
        raise ExportError(f"{spec.id} training prompt policy ID changed")
    if contract.get("system_prompt_ends_with_newline") is not True:
        raise ExportError(f"{spec.id} training system prompt must end with one newline")
    actual = contract.get("canary_rendered_prefix_sha256")
    if actual != spec.expected_rendered_prefix_sha256:
        raise ExportError(
            f"{spec.id} prompt contract changed: expected "
            f"{spec.expected_rendered_prefix_sha256}, got {actual}"
        )


def production_prompt_contract(
    spec: DeploymentSpec, target: str
) -> dict[str, str | bool]:
    """Return the exact non-retry prompt policy deployed by HowTo for *target*."""
    try:
        policy = PRODUCTION_PROMPT_POLICIES[target]
    except KeyError as exc:
        raise ExportError(f"unknown HowTo production prompt target {target!r}") from exc
    system_prompt = policy["system_prompt"]
    system_prompt_sha256 = sha256_bytes(system_prompt.encode("utf-8"))
    if system_prompt_sha256 != policy["system_prompt_sha256"]:
        raise ExportError(
            f"{target} production system prompt changed without a policy/hash update"
        )
    rendered = render_deployment_prompt(spec, system_prompt, PROMPT_CANARY_REQUEST)
    return {
        "target": target,
        "policy_id": policy["policy_id"],
        "strict_retry": False,
        "system_prompt_sha256": system_prompt_sha256,
        "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
    }


def generation_policy_contract(spec: DeploymentSpec) -> dict[str, Any]:
    """Return and authenticate the deterministic HowTo generation settings."""
    policy: dict[str, Any] = {
        "policy_id": GENERATION_POLICY_ID,
        "prompt_format_id": spec.prompt_format_id,
        "context_size": 2048,
        "max_tokens": 96,
        "temperature": 0.0,
        "top_p": None,
        "n": 1,
        "seed": 0,
        "repeat_penalty": 1.08,
        "repeat_last_n": 64,
        "add_bos": False,
        "stop_on_newline": False,
        "stop_strings": list(DUAL_STOP_TOKENS),
        "stop_token_ids": list(spec.expected_stop_ids),
    }
    canonical = json.dumps(
        policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual = sha256_bytes(canonical)
    if actual != spec.expected_generation_policy_sha256:
        raise ExportError(
            f"{spec.id} generation policy changed without a policy/hash update"
        )
    return {**policy, "policy_sha256": actual}


def qualification_split_contract() -> dict[str, Any]:
    canonical = json.dumps(
        QUALIFICATION_SPLIT_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual = sha256_bytes(canonical)
    if actual != QUALIFICATION_SPLIT_POLICY_SHA256:
        raise ExportError(
            "qualification split policy changed without a version/hash update"
        )
    return {
        "policy_id": QUALIFICATION_SPLIT_POLICY_ID,
        "policy_sha256": actual,
        **QUALIFICATION_SPLIT_POLICY,
    }


def load_adapter_tokenizer(adapter_dir: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ExportError(
            "export dependencies are missing; run `uv sync --extra export`"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ExportError("adapter tokenizer has no chat template")
    return tokenizer


def merge_adapter(
    base_snapshot: Path, adapter_dir: Path, merged_dir: Path, tokenizer: Any
) -> None:
    """Merge the LoRA on CPU and save one self-contained HF checkpoint."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ExportError(
            "export dependencies are missing; run `uv sync --extra export`"
        ) from exc

    print(f"Loading exact base snapshot {base_snapshot}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_snapshot),
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(
        model, str(adapter_dir), local_files_only=True
    )
    merged = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merged.save_pretrained(
        str(merged_dir), safe_serialization=True, max_shard_size="4GB"
    )
    tokenizer.save_pretrained(str(merged_dir))
    del peft_model
    del merged
    del model


def convert_runtime_lora(
    spec: DeploymentSpec,
    adapter_dir: Path,
    base_metadata: Path,
    destination: Path,
    toolchain: LlamaToolchain,
    base_revision: str,
) -> dict[str, Any]:
    """Export a BF16 GGUF adapter without loading or merging base weights."""
    filename = f"{spec.artifact_stem}-lora-bf16.gguf"
    output = destination / filename
    _run(
        [
            sys.executable,
            str(toolchain.lora_converter),
            "--base",
            str(base_metadata),
            "--outfile",
            str(output),
            "--outtype",
            "bf16",
            str(adapter_dir),
        ],
        env=_offline_environment(),
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise ExportError("llama.cpp did not create a non-empty BF16 GGUF LoRA")
    return {
        "file": filename,
        "kind": "runtime_lora",
        "precision": "BF16",
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "requires_base": {
            "id": spec.base_model_id,
            "format": "GGUF",
            "revision": base_revision,
        },
    }


def _offline_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def convert_and_quantize(
    spec: DeploymentSpec,
    merged_dir: Path,
    destination: Path,
    toolchain: LlamaToolchain,
    *,
    work_dir: Path,
) -> list[dict[str, Any]]:
    if toolchain.quantizer is None:
        raise ExportError("merged export requires a built llama.cpp quantizer")
    f16 = work_dir / f"{spec.artifact_stem}-f16.gguf"
    _run(
        [
            sys.executable,
            str(toolchain.converter),
            str(merged_dir),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ],
        env=_offline_environment(),
    )
    if not f16.is_file() or f16.stat().st_size == 0:
        raise ExportError("llama.cpp conversion did not create a non-empty F16 GGUF")

    artifacts: list[dict[str, Any]] = []
    for quantization in QUANTIZATIONS:
        filename = f"{spec.artifact_stem}-{quantization.lower()}.gguf"
        output = destination / filename
        _run([str(toolchain.quantizer), str(f16), str(output), quantization])
        if not output.is_file() or output.stat().st_size == 0:
            raise ExportError(f"quantizer did not create a non-empty {quantization} GGUF")
        artifacts.append(
            {
                "file": filename,
                "kind": "merged_model",
                "quantization": quantization,
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        )
    return artifacts


def build_manifest(
    *,
    spec: DeploymentSpec,
    base_revision: str,
    base_config_sha256: str,
    adapter: AdapterProvenance,
    toolchain: LlamaToolchain,
    artifacts: Iterable[dict[str, Any]],
    stop_ids: list[dict[str, int | str]],
    system_prompt: str,
    metrics_provenance: dict[str, Any] | None = None,
    export_mode: str = "all",
) -> dict[str, Any]:
    validate_stop_ids(stop_ids, spec)
    artifact_list = list(artifacts)
    format_contract = prompt_contract(spec, system_prompt)
    validate_prompt_contract(format_contract, spec)
    for artifact in artifact_list:
        artifact["model_id"] = spec.id
        artifact["display_name"] = spec.display_name
        artifact["base_revision"] = base_revision
        artifact["export_quantization"] = artifact.get(
            "quantization", artifact.get("precision")
        )
        artifact["recommended_ram_bytes_estimate"] = (
            spec.recommended_minimum_ram_bytes
        )
    evaluation_candidates: list[dict[str, Any]] = []
    lora_artifact = next(
        (item for item in artifact_list if item.get("kind") == "runtime_lora"), None
    )
    if lora_artifact is not None:
        for quantization in QUANTIZATIONS:
            evaluation_candidates.append(
                {
                    "name": f"runtime-lora-{quantization.lower()}-base",
                    "kind": "runtime_lora",
                    "adapter_file": lora_artifact["file"],
                    "base_quantization": quantization,
                    "base_revision": base_revision,
                    "base_gguf_sha256": None,
                    "recommended_ram_bytes_estimate": (
                        spec.recommended_minimum_ram_bytes
                    ),
                    "qualification_status": "required",
                }
            )
    for artifact in artifact_list:
        if artifact.get("kind") == "merged_model":
            evaluation_candidates.append(
                {
                    "name": f"merged-{artifact['quantization'].lower()}",
                    "kind": "merged_model",
                    "artifact_file": artifact["file"],
                    "recommended_ram_bytes_estimate": (
                        spec.recommended_minimum_ram_bytes
                    ),
                    "qualification_status": "required",
                }
            )
    recommended = next(
        (
            artifact
            for artifact in artifact_list
            if artifact.get("kind") == "merged_model"
            and artifact.get("quantization") == spec.recommended_quantization
        ),
        lora_artifact,
    )
    assert recommended is not None
    blocking_gates = [
        "artifact_url",
        "distribution_rights",
        "production_executable_evaluation",
        "production_macos_prompt_quality",
        "production_linux_prompt_quality",
        "production_peak_rss",
        "production_latency",
        "llama_cpp_qualification",
    ]
    if adapter.training_base_revision is None:
        blocking_gates.append("training_base_revision_attestation")
    if recommended["kind"] == "runtime_lora":
        blocking_gates.append("pinned_base_gguf")
    return {
        "schema_version": 1,
        "artifact_set": spec.artifact_stem,
        "export_mode": export_mode,
        "catalog_entry": {
            "id": spec.id,
            "tier": spec.tier,
            "display_name": spec.display_name,
            "parameters_billions": spec.parameters_billions,
            "base_model_id": spec.base_model_id,
            "base_revision": base_revision,
            "export_quantization": recommended.get(
                "quantization", recommended.get("precision")
            ),
            "artifact_kind": recommended["kind"],
            "artifact_file": recommended["file"],
            "artifact_size_bytes": recommended["size_bytes"],
            "artifact_sha256": recommended["sha256"],
            "artifact_url": None,
            "installable": False,
            "blocking_gates": blocking_gates,
            "requires_base_bundle": recommended["kind"] == "runtime_lora",
            "recommended_ram_bytes_estimate": (
                spec.recommended_minimum_ram_bytes
            ),
            "reference_bf16_metrics": {
                "executable_pass_at_1": spec.measured_executable_pass_at_1,
                "mean_latency_ms": spec.measured_mean_latency_ms,
                "peak_gpu_gb": spec.measured_peak_bf16_gb,
                "tasks": 400,
                "environment": "Transformers BF16, RTX 5090",
                "provenance": metrics_provenance,
            },
            "production_artifact_metrics": {
                "executable_pass_at_1": None,
                "mean_latency_ms": None,
                "peak_rss_bytes": None,
                "qualification_status": "required",
            },
            "stop_tokens": stop_ids,
            "chat_template_sha256": adapter.template_sha256,
            "prompt_format_id": format_contract["id"],
            "rendered_prefix_sha256": format_contract[
                "canary_rendered_prefix_sha256"
            ],
            "adapter_revision": adapter.revision,
            "training_base_revision_recorded": adapter.training_base_revision,
        },
        "model": {
            "id": spec.id,
            "tier": spec.tier,
            "display_name": spec.display_name,
            "base": {
                "id": spec.base_model_id,
                "revision": base_revision,
                "config_sha256": base_config_sha256,
            },
            "adapter": {
                "repository": adapter.remote,
                "path": spec.adapter_relative_path.as_posix(),
                "revision": adapter.revision,
                "model_sha256": adapter.model_sha256,
                "model_size_bytes": adapter.model_size,
                "tokenizer_sha256": adapter.tokenizer_sha256,
                "tokenizer_size_bytes": adapter.tokenizer_size,
                "metadata_sha256": adapter.metadata_sha256,
                "training_base_revision": adapter.training_base_revision,
            },
            "merge": (
                "peft.merge_and_unload(safe_merge=True)"
                if export_mode == "all"
                else "not_performed"
            ),
        },
        "llama_cpp": {
            "repository": LLAMA_CPP_REPOSITORY,
            "revision": toolchain.revision,
            "converter_sha256": toolchain.converter_sha256,
            "lora_converter_sha256": toolchain.lora_converter_sha256,
            "quantizer_sha256": toolchain.quantizer_sha256,
            "converter_outtype": "f16",
            "lora_converter_outtype": "bf16",
            "source_status": toolchain.source_status,
            "upstream_pr": toolchain.upstream_pr,
            "production_approved": toolchain.production_approved,
        },
        "inference_contract": {
            "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
            "prompt_format": format_contract,
            "generation_policy": generation_policy_contract(spec),
            "qualification_split_policy": qualification_split_contract(),
            "production_runtime_targets": list(PRODUCTION_RUNTIME_TARGETS),
            "production_prompt_policies": [
                production_prompt_contract(spec, target)
                for target in PRODUCTION_PROMPT_POLICIES
            ],
            "chat_template_file": "chat_template.jinja",
            "chat_template_sha256": adapter.template_sha256,
            "system_prompt_file": "system_prompt.txt",
            "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
            "add_generation_prompt": True,
            "enable_thinking": False,
            "stop_on_newline": False,
            "temperature": 0,
            "max_new_tokens": 96,
            "stop_tokens": stop_ids,
        },
        "artifacts": artifact_list,
        "evaluation_candidates": evaluation_candidates,
    }


def write_manifest(directory: Path, manifest: dict[str, Any]) -> str:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    digest = sha256_bytes(payload)
    (directory / "manifest.json").write_bytes(payload)
    (directory / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
    return digest


def export(args: argparse.Namespace) -> Path:
    spec = DEPLOYMENT_SPECS[args.model]
    repo_root = args.repo_root.resolve()
    adapter_dir = (repo_root / spec.adapter_relative_path).resolve()
    if args.output_dir is None:
        suffix = "-lora" if args.export_mode == "lora" else ""
        output_dir = (repo_root / "dist" / f"{spec.artifact_stem}{suffix}").resolve()
    else:
        output_dir = args.output_dir.resolve()
    base_revision = immutable_revision(args.base_revision, "base model revision")
    llama_revision = immutable_revision(args.llama_cpp_revision, "llama.cpp revision")

    if output_dir.exists():
        raise ExportError(
            f"output already exists: {output_dir}; move it aside rather than "
            "overwriting an immutable artifact set"
        )
    adapter = inspect_adapter(
        repo_root,
        adapter_dir,
        spec,
        expected_revision=args.adapter_revision,
        allow_downloads=args.allow_downloads,
    )
    metrics_provenance = verify_measured_metrics(
        repo_root, adapter.revision, spec
    )
    toolchain = prepare_llama_cpp(
        args.llama_cpp_dir,
        llama_revision,
        allow_downloads=args.allow_downloads,
        build_tools=not args.no_build_tools,
        need_quantizer=args.export_mode == "all",
    )
    base_metadata = resolve_base_snapshot(
        spec.base_model_id,
        base_revision,
        cache_dir=args.hf_cache_dir,
        allow_downloads=args.allow_downloads,
        metadata_only=True,
    )
    base_snapshot: Path | None = None
    if args.export_mode == "all":
        base_snapshot = resolve_base_snapshot(
            spec.base_model_id,
            base_revision,
            cache_dir=args.hf_cache_dir,
            allow_downloads=args.allow_downloads,
            metadata_only=False,
        )

    # All model loading after snapshot resolution is forced offline, even when
    # the operator permitted the preceding, explicit materialisation step.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .config import SYSTEM_PROMPT

    tokenizer = load_adapter_tokenizer(adapter_dir)
    stops = dual_stop_ids(tokenizer)
    validate_stop_ids(stops, spec)
    rendered_contract = validate_prompt_rendering(tokenizer, spec, SYSTEM_PROMPT)
    validate_prompt_contract(rendered_contract, spec)
    if args.preflight_only:
        print("Preflight passed; no merge, conversion, or quantization was run.")
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_parent = args.work_dir.resolve() if args.work_dir else output_dir.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".howto-export-work-", dir=work_parent) as work:
        work_path = Path(work)

        with tempfile.TemporaryDirectory(
            prefix=".howto-export-stage-", dir=output_dir.parent
        ) as staging:
            stage = Path(staging) / "artifact-set"
            stage.mkdir()
            artifacts = [
                convert_runtime_lora(
                    spec,
                    adapter_dir,
                    base_metadata,
                    stage,
                    toolchain,
                    base_revision,
                )
            ]
            if args.export_mode == "all":
                assert base_snapshot is not None
                merged_dir = work_path / "merged-hf"
                merged_dir.mkdir()
                merge_adapter(base_snapshot, adapter_dir, merged_dir, tokenizer)
                artifacts.extend(
                    convert_and_quantize(
                        spec, merged_dir, stage, toolchain, work_dir=work_path
                    )
                )
            shutil.copyfile(adapter_dir / "chat_template.jinja", stage / "chat_template.jinja")
            (stage / "system_prompt.txt").write_text(SYSTEM_PROMPT)
            manifest = build_manifest(
                spec=spec,
                base_revision=base_revision,
                base_config_sha256=sha256_file(base_metadata / "config.json"),
                adapter=adapter,
                toolchain=toolchain,
                artifacts=artifacts,
                stop_ids=stops,
                system_prompt=SYSTEM_PROMPT,
                metrics_provenance=metrics_provenance,
                export_mode=args.export_mode,
            )
            digest = write_manifest(stage, manifest)
            os.replace(stage, output_dir)
    print(f"Export complete: {output_dir}")
    print(f"Manifest SHA-256: {digest}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a measured HowTo deployment model as pinned GGUFs"
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEPLOYMENT_SPECS),
        default="accurate",
        help="measured Pareto tier to export (default: accurate)",
    )
    parser.add_argument(
        "--base-revision",
        required=True,
        help="exact full Hugging Face commit used for the selected model's base",
    )
    parser.add_argument(
        "--export-mode",
        choices=("lora", "all"),
        default="all",
        help=(
            "lora exports only the small BF16 runtime adapter using base metadata; "
            "all additionally downloads/requires base weights and creates merged "
            "Q4_K_M/Q5_K_M/Q6_K models"
        ),
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        required=True,
        help="clean llama.cpp checkout (created only when --allow-downloads is set)",
    )
    parser.add_argument(
        "--llama-cpp-revision",
        required=True,
        help="exact full llama.cpp commit to record and qualify",
    )
    parser.add_argument(
        "--adapter-revision",
        help="optional exact modelling-repo commit expected at HEAD",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help=(
            "permit only the explicit adapter LFS fetch, exact Hub snapshot, and "
            "pinned llama.cpp clone required by this export"
        ),
    )
    parser.add_argument(
        "--no-build-tools",
        action="store_true",
        help="require an already-built llama-quantize executable",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate/materialise inputs and toolchain without exporting weights",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        export(args)
    except ExportError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
