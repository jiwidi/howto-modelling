from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from howto.catalog import (
    AGGREGATE_PLATFORM,
    MAX_JSON_BYTES,
    _json_object,
    _positive_number,
    compose_catalog,
    qualify_entry,
    write_catalog,
)
from howto.export import (
    DEPLOYMENT_SPECS,
    ExportError,
    GENERATION_POLICY_ID,
    PRODUCTION_PROMPT_POLICIES,
    PRODUCTION_RUNTIME_TARGETS,
    QUALIFICATION_SPLIT_POLICY_ID,
    QUALIFICATION_SPLIT_POLICY_SHA256,
    QUALIFICATION_TASK_CORPUS_SHA256,
    RUNTIME_LLAMA_CPP_REVISION,
    generation_policy_contract,
    production_prompt_contract,
    qualification_split_contract,
)


LLAMA_REVISION = "c" * 40
BASE_REVISION = "d" * 40
HARNESS_REVISION = "a" * 40
TASK_SET_SHA256 = "9" * 64


def _write_qualification(root: Path, tier: str) -> Path:
    spec = DEPLOYMENT_SPECS[tier]
    model_dir = root / tier
    model_dir.mkdir()
    artifact_name = f"howto-{spec.id}-q4_k_m.gguf"
    artifact = model_dir / artifact_name
    artifact.write_bytes(f"fixture-{tier}".encode())
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "catalog_entry": {
            "id": spec.id,
            "tier": tier,
            "display_name": spec.display_name,
            "base_model_id": spec.base_model_id,
            "base_revision": BASE_REVISION,
            "training_base_revision_recorded": None,
            "export_quantization": "Q4_K_M",
            "artifact_kind": "merged_model",
            "artifact_file": artifact_name,
            "artifact_size_bytes": artifact.stat().st_size,
            "artifact_sha256": artifact_sha,
            "recommended_ram_bytes_estimate": (
                spec.recommended_minimum_ram_bytes
            ),
            "reference_bf16_metrics": {
                "executable_pass_at_1": spec.measured_executable_pass_at_1,
                "mean_latency_ms": spec.measured_mean_latency_ms,
                "peak_gpu_gb": spec.measured_peak_bf16_gb,
                "environment": "fixture",
            },
            "prompt_format_id": spec.prompt_format_id,
            "rendered_prefix_sha256": spec.expected_rendered_prefix_sha256,
            "stop_tokens": [
                {"token": "<|endoftext|>", "id": spec.expected_stop_ids[0]},
                {"token": "<|im_end|>", "id": spec.expected_stop_ids[1]},
            ],
            "chat_template_sha256": "2" * 64,
            "adapter_revision": "e" * 40,
        },
        "llama_cpp": {
            "revision": LLAMA_REVISION,
            "source_status": "unqualified",
        },
        "inference_contract": {
            "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
            "generation_policy": generation_policy_contract(spec),
            "qualification_split_policy": qualification_split_contract(),
            "production_runtime_targets": list(PRODUCTION_RUNTIME_TARGETS),
            "production_prompt_policies": [
                production_prompt_contract(spec, target)
                for target in PRODUCTION_PROMPT_POLICIES
            ],
        },
    }
    manifest_path = model_dir / "manifest.json"
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    quality_evaluations = []
    generation = generation_policy_contract(spec)
    for target, prompt_target in PRODUCTION_RUNTIME_TARGETS.items():
        prompt = production_prompt_contract(spec, prompt_target)
        report_name = f"quality-{target}.json"
        report = {
            "schema_version": 1,
            "evaluation_kind": "execution_grounded_pass_at_1",
            "model_id": spec.id,
            "artifact_sha256": artifact_sha,
            "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
            "target": target,
            "generation_policy_id": GENERATION_POLICY_ID,
            "generation_policy_sha256": generation["policy_sha256"],
            "policy_id": prompt["policy_id"],
            "strict_retry": False,
            "system_prompt_sha256": prompt["system_prompt_sha256"],
            "rendered_prompt_sha256": prompt["rendered_prompt_sha256"],
            "executable_pass_at_1": spec.measured_executable_pass_at_1,
            "task_count": 400,
            "task_corpus_sha256": QUALIFICATION_TASK_CORPUS_SHA256,
            "task_set_sha256": TASK_SET_SHA256,
            "split_policy_id": QUALIFICATION_SPLIT_POLICY_ID,
            "split_policy_sha256": QUALIFICATION_SPLIT_POLICY_SHA256,
            "harness_revision": HARNESS_REVISION,
        }
        report_payload = (json.dumps(report, sort_keys=True) + "\n").encode()
        (model_dir / report_name).write_bytes(report_payload)
        quality_evaluations.append(
            {
                "target": target,
                "policy_id": prompt["policy_id"],
                "system_prompt_sha256": prompt["system_prompt_sha256"],
                "rendered_prompt_sha256": prompt["rendered_prompt_sha256"],
                "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
                "generation_policy_id": GENERATION_POLICY_ID,
                "generation_policy_sha256": generation["policy_sha256"],
                "executable_pass_at_1": spec.measured_executable_pass_at_1,
                "task_count": 400,
                "task_corpus_sha256": QUALIFICATION_TASK_CORPUS_SHA256,
                "task_set_sha256": TASK_SET_SHA256,
                "split_policy_id": QUALIFICATION_SPLIT_POLICY_ID,
                "split_policy_sha256": QUALIFICATION_SPLIT_POLICY_SHA256,
                "harness_revision": HARNESS_REVISION,
                "report_path": report_name,
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
            }
        )
    performance_evaluations = []
    for target in PRODUCTION_RUNTIME_TARGETS:
        report_name = f"performance-{target}.json"
        report = {
            "schema_version": 1,
            "evaluation_kind": "embedded_warm_end_to_end_performance",
            "model_id": spec.id,
            "artifact_sha256": artifact_sha,
            "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
            "target": target,
            "generation_policy_id": GENERATION_POLICY_ID,
            "generation_policy_sha256": generation["policy_sha256"],
            "harness_revision": HARNESS_REVISION,
            "mean_latency_ms": 25.0,
            "peak_rss_bytes": 900_000_000,
            "platform": f"{target} fixture warm end-to-end",
        }
        report_payload = (json.dumps(report, sort_keys=True) + "\n").encode()
        (model_dir / report_name).write_bytes(report_payload)
        performance_evaluations.append(
            {
                "target": target,
                "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
                "generation_policy_id": GENERATION_POLICY_ID,
                "generation_policy_sha256": generation["policy_sha256"],
                "harness_revision": HARNESS_REVISION,
                "mean_latency_ms": 25.0,
                "peak_rss_bytes": 900_000_000,
                "platform": f"{target} fixture warm end-to-end",
                "report_path": report_name,
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
            }
        )
    qualification = {
        "model_id": spec.id,
        "manifest_path": "manifest.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_url": (
            f"https://downloads.example.test/{artifact_sha}/{artifact_name}"
        ),
        "llama_cpp_revision": LLAMA_REVISION,
        "runtime_llama_cpp_revision": RUNTIME_LLAMA_CPP_REVISION,
        "llama_cpp_qualified": True,
        "training_base_revision_attestation": {
            "base_revision": BASE_REVISION,
            "approved": True,
            "evidence": "fixture preserved training-cache record",
        },
        "distribution_rights": {
            "approved": True,
            "evidence": "fixture legal approval record",
        },
        "production_artifact_metrics": {
            "executable_pass_at_1": spec.measured_executable_pass_at_1,
            "mean_latency_ms": 25.0,
            "peak_rss_bytes": 900_000_000,
            "platform": AGGREGATE_PLATFORM,
        },
        "production_quality_evaluations": quality_evaluations,
        "production_performance_evaluations": performance_evaluations,
    }
    qualification_path = model_dir / "qualification.json"
    qualification_path.write_text(json.dumps(qualification))
    return qualification_path


class CatalogTests(unittest.TestCase):
    def test_composes_three_tiers_in_setup_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualifications = [
                _write_qualification(root, tier)
                for tier in ("accurate", "compact", "balanced")
            ]
            catalog = compose_catalog(qualifications)
            self.assertEqual(
                [entry["tier"] for entry in catalog["models"]],
                ["compact", "balanced", "accurate"],
            )
            for entry in catalog["models"]:
                self.assertEqual(entry["artifact_kind"], "merged_model")
                self.assertTrue(entry["artifact_url"].startswith("https://"))
                self.assertTrue(entry["installable"])
                self.assertGreater(entry["recommended_ram_bytes"], 0)
                self.assertEqual(
                    [
                        evaluation["target"]
                        for evaluation in entry["production_quality_evaluations"]
                    ],
                    list(PRODUCTION_RUNTIME_TARGETS),
                )
                self.assertEqual(
                    [
                        evaluation["target"]
                        for evaluation in entry[
                            "production_performance_evaluations"
                        ]
                    ],
                    list(PRODUCTION_RUNTIME_TARGETS),
                )
                self.assertEqual(
                    entry["runtime_llama_cpp_revision"],
                    RUNTIME_LLAMA_CPP_REVISION,
                )

            output = root / "release" / "model-catalog.json"
            digest = write_catalog(output, catalog)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ExportError, "refusing to overwrite"):
                write_catalog(output, catalog)

    def test_refuses_adapter_without_pinned_base_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "compact")
            manifest_path = qualification.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["catalog_entry"]["artifact_kind"] = "runtime_lora"
            payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
            manifest_path.write_bytes(payload)
            qualification_data = json.loads(qualification.read_text())
            qualification_data["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
            qualification.write_text(json.dumps(qualification_data))
            with self.assertRaisesRegex(ExportError, "without a pinned base GGUF"):
                qualify_entry(qualification)

    def test_refuses_unmerged_experimental_converter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "balanced")
            manifest_path = qualification.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["llama_cpp"]["source_status"] = "unmerged_experimental"
            payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
            manifest_path.write_bytes(payload)
            qualification_data = json.loads(qualification.read_text())
            qualification_data["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
            qualification.write_text(json.dumps(qualification_data))
            with self.assertRaisesRegex(ExportError, "unmerged experimental"):
                qualify_entry(qualification)

    def test_refuses_missing_distribution_rights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "accurate")
            data = json.loads(qualification.read_text())
            del data["distribution_rights"]
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "rights approval is required"):
                qualify_entry(qualification)

    def test_refuses_missing_training_base_revision_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "balanced")
            data = json.loads(qualification.read_text())
            del data["training_base_revision_attestation"]
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "attestation is required"):
                qualify_entry(qualification)

    def test_refuses_missing_platform_quality_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "balanced")
            data = json.loads(qualification.read_text())
            data["production_quality_evaluations"] = [
                data["production_quality_evaluations"][0]
            ]
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "exactly one record"):
                qualify_entry(qualification)

    def test_refuses_missing_platform_performance_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "balanced")
            data = json.loads(qualification.read_text())
            data["production_performance_evaluations"] = [
                data["production_performance_evaluations"][0]
            ]
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "exactly one record"):
                qualify_entry(qualification)

    def test_refuses_quality_result_for_different_deployed_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "compact")
            data = json.loads(qualification.read_text())
            data["production_quality_evaluations"][0][
                "rendered_prompt_sha256"
            ] = "0" * 64
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "wrong rendered_prompt_sha256"):
                qualify_entry(qualification)

    def test_refuses_quality_result_for_different_generation_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "compact")
            data = json.loads(qualification.read_text())
            data["production_quality_evaluations"][0][
                "generation_policy_sha256"
            ] = "0" * 64
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "wrong generation policy"):
                qualify_entry(qualification)

    def test_refuses_quality_targets_with_different_task_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "compact")
            data = json.loads(qualification.read_text())
            record = data["production_quality_evaluations"][1]
            record["task_set_sha256"] = "8" * 64
            report_path = qualification.parent / record["report_path"]
            report = json.loads(report_path.read_text())
            report["task_set_sha256"] = "8" * 64
            payload = (json.dumps(report, sort_keys=True) + "\n").encode()
            report_path.write_bytes(payload)
            record["report_sha256"] = hashlib.sha256(payload).hexdigest()
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "different held-out tasks"):
                qualify_entry(qualification)

    def test_refuses_wrong_quality_task_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "accurate")
            data = json.loads(qualification.read_text())
            data["production_quality_evaluations"][0]["task_count"] = 399
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "exactly 400 tasks"):
                qualify_entry(qualification)

    def test_refuses_quality_record_that_disagrees_with_immutable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "accurate")
            data = json.loads(qualification.read_text())
            record = data["production_quality_evaluations"][0]
            report_path = qualification.parent / record["report_path"]
            report = json.loads(report_path.read_text())
            report["artifact_sha256"] = "0" * 64
            payload = (json.dumps(report, sort_keys=True) + "\n").encode()
            report_path.write_bytes(payload)
            record["report_sha256"] = hashlib.sha256(payload).hexdigest()
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "does not match"):
                qualify_entry(qualification)

    def test_aggregate_quality_is_worst_platform_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "balanced")
            data = json.loads(qualification.read_text())
            data["production_artifact_metrics"]["executable_pass_at_1"] += 0.01
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "equal the worst"):
                qualify_entry(qualification)

    def test_aggregate_performance_is_worst_platform_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "balanced")
            data = json.loads(qualification.read_text())
            data["production_artifact_metrics"]["mean_latency_ms"] = 24.0
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "latency/RSS must equal"):
                qualify_entry(qualification)

    def test_refuses_runtime_revision_other_than_embedded_howto(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            qualification = _write_qualification(Path(directory), "compact")
            data = json.loads(qualification.read_text())
            data["runtime_llama_cpp_revision"] = "f" * 40
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "embedded HowTo revision"):
                qualify_entry(qualification)

    def test_refuses_mutable_credentialed_and_unpinned_artifact_urls(self) -> None:
        for url in (
            "https://user:secret@downloads.example.test/file.gguf",
            "https://downloads.example.test/file.gguf?token=secret",
            "https://downloads.example.test/releases/latest/file.gguf",
            "https://downloads.example.test/../file.gguf",
            "https://downloads.example.test/%2e%2e/file.gguf",
            "https://downloads.example.test/releases/stable/file.gguf",
        ):
            with tempfile.TemporaryDirectory() as directory:
                qualification = _write_qualification(Path(directory), "compact")
                data = json.loads(qualification.read_text())
                filename = json.loads(
                    (qualification.parent / "manifest.json").read_text()
                )["catalog_entry"]["artifact_file"]
                data["artifact_url"] = url.replace("file.gguf", filename)
                qualification.write_text(json.dumps(data))
                with self.assertRaises(ExportError, msg=url):
                    qualify_entry(qualification)

    def test_json_reader_rejects_duplicate_keys_nonfinite_invalid_utf8_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                b'{"a": 1, "a": 2}',
                b'{"a": NaN}',
                b'{"a": "\xff"}',
                b" " * (MAX_JSON_BYTES + 1),
            )
            for index, payload in enumerate(cases):
                path = root / f"bad-{index}.json"
                path.write_bytes(payload)
                with self.assertRaises(ExportError):
                    _json_object(path)

    def test_positive_number_rejects_nonfinite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ExportError, "positive number"):
                _positive_number(value, "fixture")

    def test_refuses_manifest_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "balanced")
            data = json.loads(qualification.read_text())
            data["manifest_path"] = "../outside.json"
            qualification.write_text(json.dumps(data))
            with self.assertRaisesRegex(ExportError, "sibling file"):
                qualify_entry(qualification)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "balanced")
            manifest = qualification.parent / "manifest.json"
            real_manifest = qualification.parent / "real-manifest.json"
            manifest.rename(real_manifest)
            manifest.symlink_to(real_manifest.name)
            with self.assertRaisesRegex(ExportError, "non-symlink"):
                qualify_entry(qualification)

    def test_refuses_symlinked_artifact_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "accurate")
            manifest = json.loads(
                (qualification.parent / "manifest.json").read_text()
            )
            artifact = qualification.parent / manifest["catalog_entry"]["artifact_file"]
            real_artifact = qualification.parent / "real-artifact.gguf"
            artifact.rename(real_artifact)
            artifact.symlink_to(real_artifact.name)
            with self.assertRaisesRegex(ExportError, "non-symlink"):
                qualify_entry(qualification)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = _write_qualification(root, "accurate")
            data = json.loads(qualification.read_text())
            report = qualification.parent / data["production_quality_evaluations"][0][
                "report_path"
            ]
            real_report = qualification.parent / "real-quality.json"
            report.rename(real_report)
            report.symlink_to(real_report.name)
            with self.assertRaisesRegex(ExportError, "non-symlink"):
                qualify_entry(qualification)

    def test_catalog_writer_refuses_dangling_output_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model-catalog.json"
            redirected = root / "redirected.json"
            output.symlink_to(redirected.name)
            with self.assertRaisesRegex(ExportError, "refusing to overwrite"):
                write_catalog(output, {"schema_version": 1})
            self.assertFalse(redirected.exists())


if __name__ == "__main__":
    unittest.main()
