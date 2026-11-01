"""Compose the immutable model catalog consumed by HowTo setup.

Export manifests are intentionally not installable on their own. This composer
requires a release-qualification record for every deployment tier, verifies the
local artifact again, and only then emits URLs and production-runtime metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

from .export import (
    DEPLOYMENT_SPECS,
    DUAL_STOP_TOKENS,
    ExportError,
    GENERATION_POLICY_ID,
    PRODUCTION_PROMPT_POLICIES,
    PRODUCTION_RUNTIME_TARGETS,
    QUALIFICATION_SPLIT_POLICY_ID,
    QUALIFICATION_SPLIT_POLICY_SHA256,
    QUALIFICATION_TASK_CORPUS_SHA256,
    REVISION_RE,
    RUNTIME_LLAMA_CPP_REVISION,
    generation_policy_contract,
    production_prompt_contract,
    qualification_split_contract,
    sha256_bytes,
    sha256_file,
)


TIER_ORDER = ("compact", "balanced", "accurate")
RAM_ROUNDING_BYTES = 256 * 1024 * 1024
QUALITY_TASK_COUNT = 400
QUALITY_REPORT_SCHEMA_VERSION = 1
QUALITY_EVALUATION_KIND = "execution_grounded_pass_at_1"
PERFORMANCE_REPORT_SCHEMA_VERSION = 1
PERFORMANCE_EVALUATION_KIND = "embedded_warm_end_to_end_performance"
MAX_JSON_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")
MAX_LATENCY_MS = 3_600_000.0
MAX_U64 = 2**64 - 1
AGGREGATE_PLATFORM = (
    "macos+linux arm64+x86_64 warm end-to-end worst-case aggregate"
)


def _reject_json_constant(value: str) -> None:
    raise ExportError(f"JSON contains non-finite number {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExportError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        if len(payload) > MAX_JSON_BYTES:
            raise ExportError(
                f"JSON input exceeds {MAX_JSON_BYTES} byte limit: {path}"
            )
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"expected a JSON object: {path}")
    return value


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _regular_sibling(owner_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{label} is required")
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.parent != Path(".")
        or relative.name in ("", ".", "..")
    ):
        raise ExportError(f"{label} must name a sibling file")
    root = owner_path.parent.resolve()
    path = root / relative.name
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ExportError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ExportError(f"{label} must be a regular non-symlink file")
    return path


def _https_url(
    value: object,
    label: str,
    *,
    filename: str,
    immutable_ids: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise ExportError(f"{label} must be an HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ExportError(f"{label} is malformed") from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExportError(
            f"{label} must be a credential-free HTTPS URL without query or fragment"
        )
    try:
        decoded_path = unquote(parsed.path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ExportError(f"{label} contains invalid path encoding") from exc
    path = PurePosixPath(decoded_path)
    if any(part in (".", "..") for part in decoded_path.split("/")):
        raise ExportError(f"{label} must not contain path traversal")
    if path.name != filename or not decoded_path.endswith(f"/{filename}"):
        raise ExportError(f"{label} filename does not match the artifact")
    lowered = decoded_path.lower()
    mutable_markers = (
        "/resolve/main/",
        "/resolve/master/",
        "/raw/main/",
        "/raw/master/",
        "/releases/latest/",
        "/latest/",
        "/refs/heads/",
    )
    if any(marker in lowered for marker in mutable_markers):
        raise ExportError(f"{label} contains a mutable branch or latest reference")
    if set(path.parts).isdisjoint(immutable_ids):
        raise ExportError(
            f"{label} must contain the artifact SHA-256 or a known immutable "
            "40-/64-hex revision as a complete path component"
        )
    return value


def _positive_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ExportError(f"{label} must be a positive number")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_U64
    ):
        raise ExportError(f"{label} must be a positive 64-bit integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ExportError(f"{label} must be a lowercase SHA-256")
    return value


def _recommended_ram(estimated: int, measured_peak_rss: int) -> int:
    with_headroom = math.ceil(measured_peak_rss * 1.25)
    rounded = math.ceil(with_headroom / RAM_ROUNDING_BYTES) * RAM_ROUNDING_BYTES
    return max(estimated, rounded)


def _evidence_report_path(
    qualification_path: Path, value: object, target: str, kind: str
) -> Path:
    return _regular_sibling(
        qualification_path, value, f"{target} {kind} report_path"
    )


def _quality_evaluations(
    qualification: dict[str, Any],
    qualification_path: Path,
    *,
    model_id: str,
    artifact_sha256: str,
    quality_floor: float,
    runtime_llama_cpp_revision: str,
) -> list[dict[str, Any]]:
    records = qualification.get("production_quality_evaluations")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ExportError(
            f"production_quality_evaluations are required for {model_id}"
        )
    by_target = {record.get("target"): record for record in records}
    expected_targets = set(PRODUCTION_RUNTIME_TARGETS)
    if len(by_target) != len(records) or set(by_target) != expected_targets:
        raise ExportError(
            f"{model_id} quality evaluations require exactly one record for "
            f"{sorted(expected_targets)}"
        )

    spec = next(item for item in DEPLOYMENT_SPECS.values() if item.id == model_id)
    expected_generation = generation_policy_contract(spec)
    qualified: list[dict[str, Any]] = []
    for target, prompt_target in PRODUCTION_RUNTIME_TARGETS.items():
        record = by_target[target]
        expected_prompt = production_prompt_contract(spec, prompt_target)
        for key in (
            "policy_id",
            "system_prompt_sha256",
            "rendered_prompt_sha256",
        ):
            if record.get(key) != expected_prompt[key]:
                raise ExportError(
                    f"{model_id} {target} quality evaluation has the wrong {key}"
                )
        if record.get("runtime_llama_cpp_revision") != runtime_llama_cpp_revision:
            raise ExportError(
                f"{model_id} {target} quality evaluation has the wrong "
                "runtime_llama_cpp_revision"
            )
        if (
            record.get("generation_policy_id") != GENERATION_POLICY_ID
            or record.get("generation_policy_sha256")
            != expected_generation["policy_sha256"]
        ):
            raise ExportError(
                f"{model_id} {target} quality evaluation has the wrong "
                "generation policy"
            )
        if record.get("task_corpus_sha256") != QUALIFICATION_TASK_CORPUS_SHA256:
            raise ExportError(
                f"{model_id} {target} quality evaluation has the wrong task corpus"
            )
        task_set_sha256 = _sha256(
            record.get("task_set_sha256"), f"{model_id} {target} task_set_sha256"
        )
        if (
            record.get("split_policy_id") != QUALIFICATION_SPLIT_POLICY_ID
            or record.get("split_policy_sha256")
            != QUALIFICATION_SPLIT_POLICY_SHA256
        ):
            raise ExportError(
                f"{model_id} {target} quality evaluation has the wrong split policy"
            )
        harness_revision = record.get("harness_revision")
        if not isinstance(harness_revision, str) or not REVISION_RE.fullmatch(
            harness_revision
        ):
            raise ExportError(
                f"{model_id} {target} quality harness revision is not immutable"
            )
        pass_at_1 = _positive_number(
            record.get("executable_pass_at_1"),
            f"{model_id} {target} executable_pass_at_1",
        )
        if pass_at_1 > 1:
            raise ExportError(
                f"{model_id} {target} executable_pass_at_1 must be <= 1"
            )
        if pass_at_1 < quality_floor:
            raise ExportError(
                f"{model_id} {target} production pass@1 {pass_at_1:.4f} is "
                f"below its qualification floor {quality_floor:.4f}"
            )
        task_count = record.get("task_count")
        if isinstance(task_count, bool) or task_count != QUALITY_TASK_COUNT:
            raise ExportError(
                f"{model_id} {target} quality evaluation must contain exactly "
                f"{QUALITY_TASK_COUNT} tasks"
            )

        report_path = _evidence_report_path(
            qualification_path, record.get("report_path"), target, "quality"
        )
        report_sha256 = record.get("report_sha256")
        if report_sha256 != sha256_file(report_path):
            raise ExportError(f"{model_id} {target} quality report checksum mismatch")
        report = _json_object(report_path)
        expected_report = {
            "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
            "evaluation_kind": QUALITY_EVALUATION_KIND,
            "model_id": model_id,
            "artifact_sha256": artifact_sha256,
            "runtime_llama_cpp_revision": runtime_llama_cpp_revision,
            "target": target,
            "generation_policy_id": GENERATION_POLICY_ID,
            "generation_policy_sha256": expected_generation["policy_sha256"],
            "policy_id": expected_prompt["policy_id"],
            "strict_retry": False,
            "system_prompt_sha256": expected_prompt["system_prompt_sha256"],
            "rendered_prompt_sha256": expected_prompt["rendered_prompt_sha256"],
            "executable_pass_at_1": pass_at_1,
            "task_count": task_count,
            "task_corpus_sha256": QUALIFICATION_TASK_CORPUS_SHA256,
            "task_set_sha256": task_set_sha256,
            "split_policy_id": QUALIFICATION_SPLIT_POLICY_ID,
            "split_policy_sha256": QUALIFICATION_SPLIT_POLICY_SHA256,
            "harness_revision": harness_revision,
        }
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ExportError(
                f"{model_id} {target} quality report does not match its "
                "qualification record or deployed prompt contract"
            )
        qualified.append(
            {
                "target": target,
                "policy_id": expected_prompt["policy_id"],
                "system_prompt_sha256": expected_prompt["system_prompt_sha256"],
                "rendered_prompt_sha256": expected_prompt[
                    "rendered_prompt_sha256"
                ],
                "runtime_llama_cpp_revision": runtime_llama_cpp_revision,
                "generation_policy_id": GENERATION_POLICY_ID,
                "generation_policy_sha256": expected_generation["policy_sha256"],
                "executable_pass_at_1": pass_at_1,
                "task_count": task_count,
                "task_corpus_sha256": QUALIFICATION_TASK_CORPUS_SHA256,
                "task_set_sha256": task_set_sha256,
                "split_policy_id": QUALIFICATION_SPLIT_POLICY_ID,
                "split_policy_sha256": QUALIFICATION_SPLIT_POLICY_SHA256,
                "harness_revision": harness_revision,
                "report_schema_version": QUALITY_REPORT_SCHEMA_VERSION,
                "report_sha256": report_sha256,
            }
        )
    if len({record["task_set_sha256"] for record in qualified}) != 1:
        raise ExportError(f"{model_id} quality targets used different held-out tasks")
    if len({record["harness_revision"] for record in qualified}) != 1:
        raise ExportError(f"{model_id} quality targets used different harness revisions")
    return qualified


def _performance_evaluations(
    qualification: dict[str, Any],
    qualification_path: Path,
    *,
    model_id: str,
    artifact_sha256: str,
    runtime_llama_cpp_revision: str,
) -> list[dict[str, Any]]:
    records = qualification.get("production_performance_evaluations")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ExportError(
            f"production_performance_evaluations are required for {model_id}"
        )
    by_target = {record.get("target"): record for record in records}
    expected_targets = set(PRODUCTION_RUNTIME_TARGETS)
    if len(by_target) != len(records) or set(by_target) != expected_targets:
        raise ExportError(
            f"{model_id} performance evaluations require exactly one record for "
            f"{sorted(expected_targets)}"
        )

    spec = next(item for item in DEPLOYMENT_SPECS.values() if item.id == model_id)
    expected_generation = generation_policy_contract(spec)
    qualified: list[dict[str, Any]] = []
    for target in PRODUCTION_RUNTIME_TARGETS:
        record = by_target[target]
        if record.get("runtime_llama_cpp_revision") != runtime_llama_cpp_revision:
            raise ExportError(
                f"{model_id} {target} performance evaluation has the wrong "
                "runtime_llama_cpp_revision"
            )
        if (
            record.get("generation_policy_id") != GENERATION_POLICY_ID
            or record.get("generation_policy_sha256")
            != expected_generation["policy_sha256"]
        ):
            raise ExportError(
                f"{model_id} {target} performance evaluation has the wrong "
                "generation policy"
            )
        harness_revision = record.get("harness_revision")
        if not isinstance(harness_revision, str) or not REVISION_RE.fullmatch(
            harness_revision
        ):
            raise ExportError(
                f"{model_id} {target} performance harness revision is not immutable"
            )
        mean_latency = _positive_number(
            record.get("mean_latency_ms"), f"{model_id} {target} mean_latency_ms"
        )
        if mean_latency > MAX_LATENCY_MS:
            raise ExportError(f"{model_id} {target} mean_latency_ms is unreasonable")
        peak_rss = _positive_int(
            record.get("peak_rss_bytes"), f"{model_id} {target} peak_rss_bytes"
        )
        platform = record.get("platform")
        if (
            not isinstance(platform, str)
            or not 1 <= len(platform) <= 200
            or not platform.isascii()
            or not platform.isprintable()
        ):
            raise ExportError(
                f"{model_id} {target} performance platform must be printable ASCII"
            )
        normalized_platform = platform.casefold()
        if (
            target not in normalized_platform
            or "warm" not in normalized_platform
            or not any(
                marker in normalized_platform
                for marker in ("end-to-end", "end to end", "e2e")
            )
        ):
            raise ExportError(
                f"{model_id} {target} performance platform must identify the "
                "target and a warm end-to-end measurement"
            )
        report_path = _evidence_report_path(
            qualification_path, record.get("report_path"), target, "performance"
        )
        report_sha256 = record.get("report_sha256")
        if report_sha256 != sha256_file(report_path):
            raise ExportError(
                f"{model_id} {target} performance report checksum mismatch"
            )
        report = _json_object(report_path)
        expected_report = {
            "schema_version": PERFORMANCE_REPORT_SCHEMA_VERSION,
            "evaluation_kind": PERFORMANCE_EVALUATION_KIND,
            "model_id": model_id,
            "artifact_sha256": artifact_sha256,
            "runtime_llama_cpp_revision": runtime_llama_cpp_revision,
            "target": target,
            "generation_policy_id": GENERATION_POLICY_ID,
            "generation_policy_sha256": expected_generation["policy_sha256"],
            "harness_revision": harness_revision,
            "mean_latency_ms": mean_latency,
            "peak_rss_bytes": peak_rss,
            "platform": platform,
        }
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ExportError(
                f"{model_id} {target} performance report does not match its "
                "qualification record"
            )
        qualified.append(
            {
                "target": target,
                "runtime_llama_cpp_revision": runtime_llama_cpp_revision,
                "generation_policy_id": GENERATION_POLICY_ID,
                "generation_policy_sha256": expected_generation["policy_sha256"],
                "mean_latency_ms": mean_latency,
                "peak_rss_bytes": peak_rss,
                "platform": platform,
                "harness_revision": harness_revision,
                "report_schema_version": PERFORMANCE_REPORT_SCHEMA_VERSION,
                "report_sha256": report_sha256,
            }
        )
    if len({record["harness_revision"] for record in qualified}) != 1:
        raise ExportError(
            f"{model_id} performance targets used different harness revisions"
        )
    return qualified


def qualify_entry(qualification_path: Path) -> dict[str, Any]:
    qualification_path = qualification_path.absolute()
    qualification_path = qualification_path.parent.resolve() / qualification_path.name
    try:
        qualification_mode = qualification_path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ExportError(f"qualification is missing: {qualification_path}") from exc
    if stat.S_ISLNK(qualification_mode) or not stat.S_ISREG(qualification_mode):
        raise ExportError("qualification must be a regular non-symlink file")
    qualification = _json_object(qualification_path)
    model_id = qualification.get("model_id")
    spec = next((item for item in DEPLOYMENT_SPECS.values() if item.id == model_id), None)
    if spec is None:
        raise ExportError(f"qualification names unknown deployment model {model_id!r}")

    manifest_path = _regular_sibling(
        qualification_path, qualification.get("manifest_path"), "manifest_path"
    )
    manifest_payload = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_payload)
    if qualification.get("manifest_sha256") != manifest_sha:
        raise ExportError(f"manifest checksum mismatch for {model_id}")
    manifest = _json_object(manifest_path)
    entry = manifest.get("catalog_entry")
    if not isinstance(entry, dict) or entry.get("id") != model_id:
        raise ExportError(f"manifest catalog entry does not match {model_id}")
    if entry.get("tier") != spec.tier:
        raise ExportError(f"manifest tier does not match {model_id}")
    if entry.get("base_model_id") != spec.base_model_id:
        raise ExportError(f"manifest base model does not match {model_id}")
    base_revision = entry.get("base_revision")
    if not isinstance(base_revision, str) or not REVISION_RE.fullmatch(base_revision):
        raise ExportError(f"manifest base revision is not immutable for {model_id}")
    adapter_revision = entry.get("adapter_revision")
    if not isinstance(adapter_revision, str) or not REVISION_RE.fullmatch(
        adapter_revision
    ):
        raise ExportError(f"manifest adapter revision is not immutable for {model_id}")
    recorded_training_revision = entry.get("training_base_revision_recorded")
    if recorded_training_revision is None:
        attestation = qualification.get("training_base_revision_attestation")
        if not isinstance(attestation, dict):
            raise ExportError(
                f"training-base revision attestation is required for {model_id}"
            )
        evidence = attestation.get("evidence")
        if (
            attestation.get("approved") is not True
            or attestation.get("base_revision") != base_revision
            or not isinstance(evidence, str)
            or not evidence.strip()
        ):
            raise ExportError(
                f"training-base revision attestation is invalid for {model_id}"
            )
        training_revision_provenance = {
            "source": "release_attestation",
            "evidence": evidence,
        }
    elif recorded_training_revision == base_revision:
        training_revision_provenance = {"source": "training_metadata"}
    else:
        raise ExportError(f"training/base revision mismatch for {model_id}")

    rights = qualification.get("distribution_rights")
    if not isinstance(rights, dict):
        raise ExportError(f"distribution-rights approval is required for {model_id}")
    rights_evidence = rights.get("evidence")
    if (
        rights.get("approved") is not True
        or not isinstance(rights_evidence, str)
        or not rights_evidence.strip()
    ):
        raise ExportError(f"distribution-rights approval is invalid for {model_id}")
    if entry.get("prompt_format_id") != spec.prompt_format_id:
        raise ExportError(f"manifest prompt format does not match {model_id}")
    if entry.get("rendered_prefix_sha256") != spec.expected_rendered_prefix_sha256:
        raise ExportError(f"manifest prompt hash does not match {model_id}")
    stop_tokens = entry.get("stop_tokens")
    if (
        not isinstance(stop_tokens, list)
        or not all(isinstance(stop, dict) for stop in stop_tokens)
        or [stop.get("token") for stop in stop_tokens] != list(DUAL_STOP_TOKENS)
    ):
        raise ExportError(f"manifest dual-stop contract is invalid for {model_id}")
    stop_ids = [stop.get("id") for stop in stop_tokens]
    if (
        any(not isinstance(stop_id, int) or stop_id < 0 for stop_id in stop_ids)
        or len(set(stop_ids)) != 2
    ):
        raise ExportError(f"manifest stop IDs are invalid for {model_id}")
    if tuple(stop_ids) != spec.expected_stop_ids:
        raise ExportError(f"manifest stop IDs do not match {model_id}")
    if entry.get("artifact_kind") != "merged_model":
        raise ExportError(
            f"{model_id} recommends an adapter without a pinned base GGUF; "
            "setup catalogs currently require one merged artifact"
        )
    if entry.get("export_quantization") != spec.recommended_quantization:
        raise ExportError(
            f"{model_id} setup artifact must be {spec.recommended_quantization}"
        )
    reference = entry.get("reference_bf16_metrics")
    expected_reference = {
        "executable_pass_at_1": spec.measured_executable_pass_at_1,
        "mean_latency_ms": spec.measured_mean_latency_ms,
        "peak_gpu_gb": spec.measured_peak_bf16_gb,
    }
    if not isinstance(reference, dict) or any(
        reference.get(key) != value for key, value in expected_reference.items()
    ):
        raise ExportError(f"reference BF16 metrics do not match {model_id}")

    llama_cpp = manifest.get("llama_cpp")
    if not isinstance(llama_cpp, dict):
        raise ExportError(f"manifest has no llama.cpp provenance for {model_id}")
    if llama_cpp.get("source_status") == "unmerged_experimental":
        raise ExportError(
            f"{model_id} was exported with an unmerged experimental llama.cpp PR"
        )
    if qualification.get("llama_cpp_revision") != llama_cpp.get("revision"):
        raise ExportError(f"llama.cpp revision mismatch for {model_id}")
    converter_llama_cpp_revision = llama_cpp.get("revision")
    if not isinstance(converter_llama_cpp_revision, str) or not REVISION_RE.fullmatch(
        converter_llama_cpp_revision
    ):
        raise ExportError(f"llama.cpp export revision is not immutable for {model_id}")
    if qualification.get("llama_cpp_qualified") is not True:
        raise ExportError(f"llama.cpp qualification is required for {model_id}")
    runtime_llama_cpp_revision = qualification.get("runtime_llama_cpp_revision")
    if runtime_llama_cpp_revision != RUNTIME_LLAMA_CPP_REVISION:
        raise ExportError(
            f"{model_id} runtime_llama_cpp_revision must be the embedded HowTo "
            f"revision {RUNTIME_LLAMA_CPP_REVISION}"
        )
    inference_contract = manifest.get("inference_contract")
    if (
        not isinstance(inference_contract, dict)
        or inference_contract.get("runtime_llama_cpp_revision")
        != runtime_llama_cpp_revision
    ):
        raise ExportError(
            f"manifest runtime llama.cpp revision does not match {model_id}"
        )
    if inference_contract.get("generation_policy") != generation_policy_contract(spec):
        raise ExportError(f"manifest generation policy does not match {model_id}")
    if (
        inference_contract.get("qualification_split_policy")
        != qualification_split_contract()
    ):
        raise ExportError(f"manifest qualification split policy does not match {model_id}")
    if inference_contract.get("production_runtime_targets") != list(
        PRODUCTION_RUNTIME_TARGETS
    ):
        raise ExportError(f"manifest production runtime targets do not match {model_id}")
    expected_prompt_policies = [
        production_prompt_contract(spec, target)
        for target in PRODUCTION_PROMPT_POLICIES
    ]
    if inference_contract.get("production_prompt_policies") != expected_prompt_policies:
        raise ExportError(f"manifest production prompt policies do not match {model_id}")

    artifact_name = entry.get("artifact_file")
    if (
        not isinstance(artifact_name, str)
        or ARTIFACT_FILENAME_RE.fullmatch(artifact_name) is None
        or Path(artifact_name).name != artifact_name
    ):
        raise ExportError(f"invalid artifact filename for {model_id}")
    artifact_path = _regular_sibling(
        manifest_path, artifact_name, f"artifact for {model_id}"
    )
    artifact_size = artifact_path.stat().st_size
    artifact_sha = sha256_file(artifact_path)
    if entry.get("artifact_size_bytes") != artifact_size:
        raise ExportError(f"artifact size mismatch for {model_id}")
    if entry.get("artifact_sha256") != artifact_sha:
        raise ExportError(f"artifact checksum mismatch for {model_id}")

    metrics = qualification.get("production_artifact_metrics")
    if not isinstance(metrics, dict):
        raise ExportError(f"production_artifact_metrics are required for {model_id}")
    pass_at_1 = _positive_number(
        metrics.get("executable_pass_at_1"), f"{model_id} executable_pass_at_1"
    )
    if pass_at_1 > 1:
        raise ExportError(f"{model_id} executable_pass_at_1 must be <= 1")
    quality_floor = spec.measured_executable_pass_at_1 - 0.03
    if pass_at_1 < quality_floor:
        raise ExportError(
            f"{model_id} production pass@1 {pass_at_1:.4f} is below its "
            f"qualification floor {quality_floor:.4f}"
        )
    quality_evaluations = _quality_evaluations(
        qualification,
        qualification_path,
        model_id=model_id,
        artifact_sha256=artifact_sha,
        quality_floor=quality_floor,
        runtime_llama_cpp_revision=runtime_llama_cpp_revision,
    )
    worst_target_pass_at_1 = min(
        evaluation["executable_pass_at_1"] for evaluation in quality_evaluations
    )
    if not math.isclose(
        pass_at_1, worst_target_pass_at_1, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ExportError(
            f"{model_id} production_artifact_metrics executable_pass_at_1 must "
            "equal the worst macOS/Linux quality result"
        )
    mean_latency = _positive_number(
        metrics.get("mean_latency_ms"), f"{model_id} mean_latency_ms"
    )
    peak_rss = _positive_int(
        metrics.get("peak_rss_bytes"), f"{model_id} peak_rss_bytes"
    )
    platform = metrics.get("platform")
    performance_evaluations = _performance_evaluations(
        qualification,
        qualification_path,
        model_id=model_id,
        artifact_sha256=artifact_sha,
        runtime_llama_cpp_revision=runtime_llama_cpp_revision,
    )
    worst_target_latency = max(
        evaluation["mean_latency_ms"] for evaluation in performance_evaluations
    )
    worst_target_rss = max(
        evaluation["peak_rss_bytes"] for evaluation in performance_evaluations
    )
    if not math.isclose(
        mean_latency, worst_target_latency, rel_tol=0.0, abs_tol=1e-12
    ) or peak_rss != worst_target_rss:
        raise ExportError(
            f"{model_id} production_artifact_metrics latency/RSS must equal the "
            "worst macOS/Linux performance results"
        )
    if platform != AGGREGATE_PLATFORM:
        raise ExportError(
            f"{model_id} production aggregate platform must be {AGGREGATE_PLATFORM!r}"
        )

    artifact_url = _https_url(
        qualification.get("artifact_url"),
        f"{model_id} artifact_url",
        filename=artifact_name,
        immutable_ids=(
            artifact_sha,
            base_revision,
            adapter_revision,
            converter_llama_cpp_revision,
        ),
    )

    return {
        "id": model_id,
        "tier": spec.tier,
        "display_name": spec.display_name,
        "parameters_billions": spec.parameters_billions,
        "base_model_id": entry["base_model_id"],
        "base_revision": entry["base_revision"],
        "training_base_revision_provenance": training_revision_provenance,
        "distribution_rights_evidence": rights_evidence,
        "export_quantization": entry["export_quantization"],
        "artifact_kind": "merged_model",
        "artifact_file": artifact_name,
        "artifact_size_bytes": artifact_size,
        "artifact_sha256": artifact_sha,
        "artifact_url": artifact_url,
        "installable": True,
        "recommended_ram_bytes": _recommended_ram(
            int(entry["recommended_ram_bytes_estimate"]), peak_rss
        ),
        "reference_bf16_metrics": entry["reference_bf16_metrics"],
        "production_artifact_metrics": {
            "executable_pass_at_1": pass_at_1,
            "mean_latency_ms": mean_latency,
            "peak_rss_bytes": peak_rss,
            "platform": platform,
            "qualification_report_sha256": sha256_file(qualification_path),
        },
        "production_quality_evaluations": quality_evaluations,
        "production_performance_evaluations": performance_evaluations,
        "prompt_format_id": entry["prompt_format_id"],
        "rendered_prefix_sha256": entry["rendered_prefix_sha256"],
        "stop_tokens": entry["stop_tokens"],
        "chat_template_sha256": entry["chat_template_sha256"],
        "adapter_revision": adapter_revision,
        "llama_cpp_revision": converter_llama_cpp_revision,
        "runtime_llama_cpp_revision": runtime_llama_cpp_revision,
        "manifest_sha256": manifest_sha,
    }


def compose_catalog(qualification_paths: Sequence[Path]) -> dict[str, Any]:
    entries = [qualify_entry(path) for path in qualification_paths]
    by_tier = {entry["tier"]: entry for entry in entries}
    if len(by_tier) != len(entries):
        raise ExportError("release qualifications contain duplicate model tiers")
    missing = set(TIER_ORDER) - set(by_tier)
    extra = set(by_tier) - set(TIER_ORDER)
    if missing or extra:
        raise ExportError(
            f"release catalog requires exactly {TIER_ORDER}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return {
        "schema_version": 1,
        "qualified": True,
        "models": [by_tier[tier] for tier in TIER_ORDER],
    }


def write_catalog(output: Path, catalog: dict[str, Any]) -> str:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    checksum = output.with_suffix(output.suffix + ".sha256")
    if _path_entry_exists(output) or _path_entry_exists(checksum):
        raise ExportError(f"refusing to overwrite immutable catalog: {output}")
    payload = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    checksum_payload = f"{digest}  {output.name}\n".encode("utf-8")
    _write_new_atomic(output, payload)
    try:
        _write_new_atomic(checksum, checksum_payload)
    except Exception:
        # This file was linked by this call and cannot predate it because of the
        # create-new check above. Avoid leaving a catalog without its checksum.
        output.unlink(missing_ok=True)
        raise
    return digest


def _write_new_atomic(path: Path, payload: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ExportError(f"refusing to overwrite immutable file: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose a qualified, immutable HowTo setup model catalog"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "qualification",
        nargs="+",
        type=Path,
        help="three qualification JSON files (compact, balanced, accurate)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = compose_catalog(args.qualification)
        digest = write_catalog(args.output, catalog)
    except (ExportError, OSError) as exc:
        parser.error(str(exc))
    print(f"Catalog: {args.output.resolve()}")
    print(f"Catalog SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
