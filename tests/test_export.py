from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from howto.export import (
    AdapterProvenance,
    DEPLOYMENT_SPECS,
    ExportError,
    LfsPointer,
    LlamaToolchain,
    PRODUCTION_PROMPT_POLICIES,
    PRODUCTION_RUNTIME_TARGETS,
    QUALIFICATION_SPLIT_POLICY_SHA256,
    QWEN35_LORA_PR_REVISION,
    RUNTIME_LLAMA_CPP_REVISION,
    TRAINING_PROMPT_POLICY_ID,
    TRAINING_SYSTEM_PROMPT_SHA256,
    _materialize_lfs_files,
    _verify_lfs_object,
    build_manifest,
    convert_runtime_lora,
    dual_stop_ids,
    generation_policy_contract,
    immutable_revision,
    llama_cpp_source_provenance,
    parse_lfs_pointer,
    prompt_contract,
    production_prompt_contract,
    qualification_split_contract,
    render_deployment_prompt,
    resolve_base_snapshot,
    validate_prompt_rendering,
    validate_prompt_contract,
    validate_stop_ids,
    verify_measured_metrics,
    write_manifest,
)
from howto.prompts import TRAINING_SYSTEM_PROMPT as SYSTEM_PROMPT


REVISION = "a" * 40


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self, ids: dict[str, int]):
        self.ids = ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, self.unk_token_id)


class ExportTests(unittest.TestCase):
    def test_catalog_contains_only_three_measured_pareto_tiers(self) -> None:
        self.assertEqual(set(DEPLOYMENT_SPECS), {"compact", "balanced", "accurate"})
        self.assertEqual(
            [
                DEPLOYMENT_SPECS[tier].measured_executable_pass_at_1
                for tier in ("compact", "balanced", "accurate")
            ],
            [0.6375, 0.6775, 0.745],
        )
        self.assertEqual(
            len({spec.adapter_relative_path for spec in DEPLOYMENT_SPECS.values()}),
            3,
        )

    def test_prompt_formats_are_family_specific_and_hashed(self) -> None:
        compact = DEPLOYMENT_SPECS["compact"]
        balanced = DEPLOYMENT_SPECS["balanced"]
        compact_contract = prompt_contract(compact, "system\n")
        balanced_contract = prompt_contract(balanced, "system\n")
        self.assertEqual(compact_contract["id"], "qwen_chatml_no_thinking")
        self.assertEqual(balanced_contract["id"], "qwen35_no_thinking")
        self.assertNotEqual(
            compact_contract["canary_rendered_prefix_sha256"],
            balanced_contract["canary_rendered_prefix_sha256"],
        )
        self.assertTrue(
            balanced_contract["assistant_generation_prefix"].endswith(
                "<think>\n\n</think>\n\n"
            )
        )
        self.assertFalse(balanced_contract["stop_on_newline"])

    def test_qwen35_prompt_validation_rejects_default_thinking_prefix(self) -> None:
        spec = DEPLOYMENT_SPECS["balanced"]
        system = "system\n"
        expected = render_deployment_prompt(spec, system, "__HOWTO_USER_REQUEST__")
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = expected
        contract = validate_prompt_rendering(tokenizer, spec, system)
        self.assertEqual(contract["id"], "qwen35_no_thinking")
        self.assertFalse(contract["stop_on_newline"])
        self.assertFalse(
            tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
        )
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>assistant\n<think>\n"
        )
        with self.assertRaisesRegex(ExportError, "no longer matches"):
            validate_prompt_rendering(tokenizer, spec, system)

    def test_deployment_prompt_hash_is_pinned_to_training_contract(self) -> None:
        spec = DEPLOYMENT_SPECS["accurate"]
        contract = prompt_contract(spec, SYSTEM_PROMPT)
        validate_prompt_contract(contract, spec)
        with self.assertRaisesRegex(ExportError, "training system prompt changed"):
            validate_prompt_contract(prompt_contract(spec, "changed"), spec)

    def test_training_prompt_policy_pins_the_final_newline(self) -> None:
        spec = DEPLOYMENT_SPECS["balanced"]
        self.assertTrue(SYSTEM_PROMPT.endswith("\n"))
        self.assertFalse(SYSTEM_PROMPT.endswith("\n\n"))
        self.assertEqual(
            hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            TRAINING_SYSTEM_PROMPT_SHA256,
        )
        contract = prompt_contract(spec, SYSTEM_PROMPT)
        self.assertEqual(
            contract["system_prompt_policy_id"], TRAINING_PROMPT_POLICY_ID
        )
        with self.assertRaisesRegex(ExportError, "final newline"):
            validate_prompt_contract(prompt_contract(spec, SYSTEM_PROMPT.rstrip()), spec)

    def test_production_prompt_contracts_pin_both_non_retry_targets(self) -> None:
        expected_system_hashes = {
            "macos": "a31c2650c55e6aad3c54d81fb7430601ee4dfd5a3f023ca31ae43e8c6d733ab8",
            "linux": "d17f6a13075035601f90dbd8d0943da39ae8fb5b328a4d7dee2b27e33b789cc9",
        }
        expected_rendered_hashes = {
            ("compact", "macos"): (
                "0690fa0611e4c849adabffbb82cd58c664589b6e63fe25c99177d43709c8b805"
            ),
            ("compact", "linux"): (
                "f33c6f6813dd35e64fe88bac19f64da5136e96afdd0c777b9e8b22503ea19cd1"
            ),
            ("balanced", "macos"): (
                "1ecf0eaf474f317d71648840c12295786f49612b6b3e266f2d5c2d1c0478045f"
            ),
            ("balanced", "linux"): (
                "fcd1737e211052f735b11045b503646c702567030b41ca263f18e51a54b54309"
            ),
            ("accurate", "macos"): (
                "1ecf0eaf474f317d71648840c12295786f49612b6b3e266f2d5c2d1c0478045f"
            ),
            ("accurate", "linux"): (
                "fcd1737e211052f735b11045b503646c702567030b41ca263f18e51a54b54309"
            ),
        }
        contracts = {
            (tier, target): production_prompt_contract(spec, target)
            for tier, spec in DEPLOYMENT_SPECS.items()
            for target in PRODUCTION_PROMPT_POLICIES
        }
        self.assertTrue(
            all(
                policy["system_prompt"].endswith("\n")
                and not policy["system_prompt"].endswith("\n\n")
                for policy in PRODUCTION_PROMPT_POLICIES.values()
            )
        )
        self.assertEqual(
            contracts[("compact", "macos")]["policy_id"],
            "howto.command-generation.macos.non-retry.v1",
        )
        self.assertEqual(
            contracts[("compact", "linux")]["policy_id"],
            "howto.command-generation.linux.non-retry.v1",
        )
        for (tier, target), contract in contracts.items():
            self.assertFalse(contract["strict_retry"])
            self.assertEqual(
                contract["system_prompt_sha256"], expected_system_hashes[target]
            )
            self.assertEqual(
                contract["rendered_prompt_sha256"],
                expected_rendered_hashes[(tier, target)],
            )

    def test_generation_and_split_policies_are_versioned_and_hashed(self) -> None:
        expected_generation_hashes = {
            "compact": "bf5cc2fc03580a33d0a71f19153a0f52ed35715adcddb1277e2326b77afbf35e",
            "balanced": "65b4d051714ee98d3c1e405af9e84b77230637e188fbe66695135ff0673a4660",
            "accurate": "65b4d051714ee98d3c1e405af9e84b77230637e188fbe66695135ff0673a4660",
        }
        for tier, expected in expected_generation_hashes.items():
            contract = generation_policy_contract(DEPLOYMENT_SPECS[tier])
            self.assertEqual(contract["policy_sha256"], expected)
            self.assertEqual(contract["temperature"], 0.0)
            self.assertIsNone(contract["top_p"])
            self.assertFalse(contract["add_bos"])
        self.assertEqual(
            qualification_split_contract()["policy_sha256"],
            QUALIFICATION_SPLIT_POLICY_SHA256,
        )
        self.assertEqual(
            list(PRODUCTION_RUNTIME_TARGETS),
            ["macos-arm64", "macos-x86_64", "linux-arm64", "linux-x86_64"],
        )

    def test_parses_lfs_pointer(self) -> None:
        pointer = parse_lfs_pointer(
            b"version https://git-lfs.github.com/spec/v1\n"
            + b"oid sha256:"
            + b"1" * 64
            + b"\nsize 1234\n"
        )
        self.assertEqual(pointer, LfsPointer(sha256="1" * 64, size=1234))
        self.assertIsNone(parse_lfs_pointer(b"real model bytes"))

    def test_offline_lfs_refusal_never_invokes_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pointer_path = root / "adapter.safetensors"
            pointer_path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{'2' * 64}\n"
                "size 999\n"
            )
            runner = Mock()
            with self.assertRaisesRegex(ExportError, "No download was attempted"):
                _materialize_lfs_files(
                    root,
                    [Path("adapter.safetensors")],
                    allow_downloads=False,
                    runner=runner,
                )
            runner.assert_not_called()

    def test_materialized_lfs_object_must_match_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            payload = b"fixture adapter, not a model"
            path.write_bytes(payload)
            expected = LfsPointer(
                sha256=hashlib.sha256(payload).hexdigest(), size=len(payload)
            )
            _verify_lfs_object(path, expected)
            with self.assertRaisesRegex(ExportError, "checksum mismatch"):
                _verify_lfs_object(
                    path, LfsPointer(sha256="0" * 64, size=len(payload))
                )

    def test_revisions_must_be_full_immutable_hashes(self) -> None:
        self.assertEqual(immutable_revision(REVISION, "base"), REVISION)
        with self.assertRaisesRegex(ExportError, "full immutable"):
            immutable_revision("main", "base")
        with self.assertRaisesRegex(ExportError, "full immutable"):
            immutable_revision("abc1234", "base")

    def test_qwen35_lora_fix_is_recorded_as_unmerged_not_production(self) -> None:
        provenance = llama_cpp_source_provenance(QWEN35_LORA_PR_REVISION)
        self.assertEqual(provenance["source_status"], "unmerged_experimental")
        self.assertEqual(
            provenance["upstream_pr"],
            "https://github.com/ggml-org/llama.cpp/pull/24627",
        )
        self.assertFalse(provenance["production_approved"])

    def test_base_resolution_is_local_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            downloader = Mock(return_value=str(snapshot))
            resolved = resolve_base_snapshot(
                "Qwen/Qwen3.5-2B",
                REVISION,
                cache_dir=None,
                allow_downloads=False,
                metadata_only=True,
                downloader=downloader,
            )
            self.assertEqual(resolved, snapshot.resolve())
            self.assertTrue(downloader.call_args.kwargs["local_files_only"])
            self.assertEqual(downloader.call_args.kwargs["revision"], REVISION)

    def test_download_flag_is_forwarded_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            downloader = Mock(return_value=str(snapshot))
            resolve_base_snapshot(
                "Qwen/Qwen3.5-2B",
                REVISION,
                cache_dir=Path(directory) / "cache",
                allow_downloads=True,
                metadata_only=True,
                downloader=downloader,
            )
            self.assertFalse(downloader.call_args.kwargs["local_files_only"])
            patterns = downloader.call_args.kwargs["allow_patterns"]
            self.assertNotIn("*.safetensors", patterns)
            self.assertNotIn("*.bin", patterns)

    def test_full_snapshot_without_weights_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            downloader = Mock(return_value=str(snapshot))
            with self.assertRaisesRegex(ExportError, "no model weights"):
                resolve_base_snapshot(
                    "Qwen/Qwen3.5-2B",
                    REVISION,
                    cache_dir=None,
                    allow_downloads=False,
                    metadata_only=False,
                    downloader=downloader,
                )

    def test_dual_stop_ids_are_both_required_and_distinct(self) -> None:
        tokenizer = FakeTokenizer({"<|endoftext|>": 101, "<|im_end|>": 102})
        self.assertEqual(
            dual_stop_ids(tokenizer),
            [
                {"token": "<|endoftext|>", "id": 101},
                {"token": "<|im_end|>", "id": 102},
            ],
        )
        with self.assertRaisesRegex(ExportError, "same id"):
            dual_stop_ids(
                FakeTokenizer({"<|endoftext|>": 101, "<|im_end|>": 101})
            )
        with self.assertRaisesRegex(ExportError, "does not define"):
            dual_stop_ids(FakeTokenizer({"<|endoftext|>": 101}))

    def test_each_tier_pins_its_tokenizer_stop_ids(self) -> None:
        compact = DEPLOYMENT_SPECS["compact"]
        accurate = DEPLOYMENT_SPECS["accurate"]
        validate_stop_ids(
            [
                {"token": "<|endoftext|>", "id": 151643},
                {"token": "<|im_end|>", "id": 151645},
            ],
            compact,
        )
        with self.assertRaisesRegex(ExportError, "stop IDs changed"):
            validate_stop_ids(
                [
                    {"token": "<|endoftext|>", "id": 151643},
                    {"token": "<|im_end|>", "id": 151645},
                ],
                accurate,
            )

    def test_runtime_lora_command_uses_metadata_and_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            metadata = root / "base-metadata"
            output = root / "output"
            adapter.mkdir()
            metadata.mkdir()
            output.mkdir()
            toolchain = LlamaToolchain(
                revision="c" * 40,
                converter=Path("convert_hf_to_gguf.py"),
                lora_converter=Path("convert_lora_to_gguf.py"),
                quantizer=None,
                converter_sha256="5" * 64,
                lora_converter_sha256="6" * 64,
                quantizer_sha256=None,
            )

            def create_fixture(argv: list[str], **_: object) -> None:
                outfile = Path(argv[argv.index("--outfile") + 1])
                outfile.write_bytes(b"fixture gguf lora")

            with patch("howto.export._run", side_effect=create_fixture) as runner:
                artifact = convert_runtime_lora(
                    DEPLOYMENT_SPECS["accurate"],
                    adapter,
                    metadata,
                    output,
                    toolchain,
                    REVISION,
                )
            argv = runner.call_args.args[0]
            self.assertEqual(argv[argv.index("--base") + 1], str(metadata))
            self.assertEqual(argv[argv.index("--outtype") + 1], "bf16")
            self.assertEqual(argv[-1], str(adapter))
            self.assertEqual(artifact["kind"], "runtime_lora")
            self.assertEqual(artifact["requires_base"]["revision"], REVISION)

    def test_manifest_is_canonical_and_self_authenticated(self) -> None:
        adapter = AdapterProvenance(
            revision="b" * 40,
            remote="git@example.invalid:org/repo.git",
            model_sha256="1" * 64,
            model_size=10,
            tokenizer_sha256="2" * 64,
            tokenizer_size=20,
            template_sha256="3" * 64,
            metadata_sha256={"run_meta.json": "4" * 64},
        )
        toolchain = LlamaToolchain(
            revision="c" * 40,
            converter=Path("convert_hf_to_gguf.py"),
            lora_converter=Path("convert_lora_to_gguf.py"),
            quantizer=None,
            converter_sha256="5" * 64,
            lora_converter_sha256="6" * 64,
            quantizer_sha256=None,
        )
        artifacts = [
            {
                "file": "howto-qwen35-2b-lora-bf16.gguf",
                "kind": "runtime_lora",
                "precision": "BF16",
                "sha256": "7" * 64,
                "size_bytes": 30,
            }
        ]
        manifest = build_manifest(
            spec=DEPLOYMENT_SPECS["accurate"],
            base_revision=REVISION,
            base_config_sha256="8" * 64,
            adapter=adapter,
            toolchain=toolchain,
            artifacts=artifacts,
            stop_ids=[
                {"token": "<|endoftext|>", "id": 248044},
                {"token": "<|im_end|>", "id": 248046},
            ],
            system_prompt=SYSTEM_PROMPT,
            export_mode="lora",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_manifest(root, manifest)
            payload = (root / "manifest.json").read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            self.assertEqual(
                (root / "manifest.sha256").read_text(),
                f"{digest}  manifest.json\n",
            )
            decoded = json.loads(payload)
            self.assertEqual(decoded["model"]["base"]["revision"], REVISION)
            self.assertEqual(len(decoded["inference_contract"]["stop_tokens"]), 2)
            self.assertEqual(
                decoded["inference_contract"]["runtime_llama_cpp_revision"],
                RUNTIME_LLAMA_CPP_REVISION,
            )
            self.assertEqual(
                decoded["inference_contract"]["generation_policy"][
                    "policy_sha256"
                ],
                generation_policy_contract(DEPLOYMENT_SPECS["accurate"])[
                    "policy_sha256"
                ],
            )
            self.assertEqual(
                decoded["inference_contract"]["qualification_split_policy"][
                    "policy_sha256"
                ],
                QUALIFICATION_SPLIT_POLICY_SHA256,
            )
            self.assertEqual(
                decoded["inference_contract"]["production_runtime_targets"],
                list(PRODUCTION_RUNTIME_TARGETS),
            )
            self.assertEqual(
                {
                    policy["target"]
                    for policy in decoded["inference_contract"][
                        "production_prompt_policies"
                    ]
                },
                {"macos", "linux"},
            )
            self.assertFalse(decoded["catalog_entry"]["installable"])
            self.assertTrue(decoded["catalog_entry"]["requires_base_bundle"])
            self.assertIn(
                "distribution_rights",
                decoded["catalog_entry"]["blocking_gates"],
            )
            self.assertIn(
                "training_base_revision_attestation",
                decoded["catalog_entry"]["blocking_gates"],
            )
            self.assertIn(
                "production_macos_prompt_quality",
                decoded["catalog_entry"]["blocking_gates"],
            )
            self.assertIn(
                "production_linux_prompt_quality",
                decoded["catalog_entry"]["blocking_gates"],
            )
            self.assertIsNone(
                decoded["catalog_entry"]["production_artifact_metrics"][
                    "peak_rss_bytes"
                ]
            )
            self.assertEqual(
                decoded["catalog_entry"]["reference_bf16_metrics"][
                    "executable_pass_at_1"
                ],
                0.745,
            )
            self.assertEqual(
                decoded["evaluation_candidates"][0]["kind"], "runtime_lora"
            )
            self.assertEqual(
                {
                    candidate["base_quantization"]
                    for candidate in decoded["evaluation_candidates"]
                },
                {"Q4_K_M", "Q5_K_M", "Q6_K"},
            )

    def test_catalog_metrics_are_checked_against_reports(self) -> None:
        spec = DEPLOYMENT_SPECS["compact"]
        evaluation = json.dumps(
            [{"model": f"{spec.id}-sft", "pass@1": 0.6375, "n": 400}]
        ).encode()
        latency = json.dumps(
            [
                {
                    "model": spec.id,
                    "tuned": True,
                    "variant": "sft",
                    "mean_ms": 86.4,
                    "peak_gpu_gb": 1.14,
                }
            ]
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            (reports / "evaluation.json").write_bytes(evaluation)
            (reports / "latency.json").write_bytes(latency)

            def committed(_: Path, __: str, relative: Path) -> bytes:
                return (root / relative).read_bytes()

            with patch("howto.export._git_blob", side_effect=committed):
                provenance = verify_measured_metrics(root, "a" * 40, spec)
            self.assertEqual(provenance["repository_revision"], "a" * 40)
            self.assertEqual(
                provenance["evaluation"]["sha256"], hashlib.sha256(evaluation).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
