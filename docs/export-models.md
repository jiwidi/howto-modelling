# Production model export and setup catalog

The deployment exporter intentionally exposes only the three measured Pareto
tiers. It does not offer every experiment to HowTo setup.

| tier | tuned model | BF16 executable pass@1 | BF16 mean | BF16 peak GPU | setup RAM estimate* |
|---|---|---:|---:|---:|---:|
| Compact | Qwen2.5 Coder 0.5B | 63.75% | 86.4 ms | 1.14 GB | 1.5 GiB |
| Balanced | Qwen3.5 0.8B | 67.75% | 180.0 ms | 1.62 GB | 2 GiB |
| Accurate | Qwen3.5 2B | 74.50% | 181.8 ms | 3.96 GB | 4 GiB |

\*The setup number is a conservative pre-qualification estimate, not a renamed
GPU measurement. The final setup catalog derives its recommendation from
measured embedded-runtime peak RSS with 25% headroom. Artifact disk size and
SHA-256 are taken from the generated file, never estimated.

## Two export lanes

`--export-mode lora` creates a small BF16 GGUF LoRA from adapter weights plus
the exact base config/tokenizer metadata. It does not need base weights. This is
the quickest way to compare a pinned quantized base plus a precise adapter, but
it is not independently installable and cannot enter the setup catalog without
the matching base GGUF.

`--export-mode all` additionally merges the LoRA into the exact Hugging Face
base and emits `Q4_K_M`, `Q5_K_M`, and `Q6_K` GGUF candidates. Its recommended
setup artifact is the single-file merged `Q4_K_M`; all candidates still require
embedded-runtime qualification.

The Qwen2.5 Compact tier uses the same PEFT/llama.cpp conversion pipeline as the
Qwen3.5 tiers. It keeps its own 2,507-byte ChatML template. Balanced and
Accurate share the separate 7,755-byte Qwen3.5 template; no template is copied
between model families.

## Offline and provenance guarantees

The default is local-only, including `uv` dependency resolution in the Make
targets. The command refuses a missing base, missing
llama.cpp checkout, or an adapter/tokenizer that is still a Git LFS pointer. It
does not initiate a network request. `ALLOW_DOWNLOADS=1` is the one explicit
opt-in for a targeted adapter LFS pull, exact Hub snapshot, and exact llama.cpp
clone (and for `uv` to materialize missing locked dependencies). LoRA-only Hub
downloads use a metadata allow-list that excludes model weight filenames.

Adapter payloads expected by the three tiers are:

| tier | adapter bytes | tokenizer bytes |
|---|---:|---:|
| Compact | 70,430,032 | 11,421,892 |
| Balanced | 51,143,344 | 19,989,325 |
| Accurate | 87,319,256 | 19,989,325 |

Every base and llama.cpp revision must be a full immutable commit hash. The
training run recorded each base model ID but not its revision. Recover that
commit from preserved training-host records; if it cannot be established,
retrain with revision capture before distributing derived weights. Do not use
today's Hub `main` as a guess.

The exporter can smoke-test the approved-but-unmerged Qwen3.5 LoRA fix from
[llama.cpp PR #24627](https://github.com/ggml-org/llama.cpp/pull/24627) at exact
commit `f839835a3401f0bf000d362dc10ba6b1c50d3a3f`. Manifests label that checkout
`unmerged_experimental`, and the setup-catalog composer refuses it. A normal
model-loading smoke test is not sufficient qualification for Qwen3.5 LoRA
V-head conversion.

## Prompt contract

Compact uses prompt format `qwen_chatml_no_thinking`. Balanced and Accurate use
`qwen35_no_thinking`, whose assistant prefill is exactly:

```text
<|im_start|>assistant
<think>

</think>

```

This prefill is required. Default GGUF chat rendering ends at the assistant
newline and makes Qwen3.5 generate `<think>` itself; a newline stop then cuts
the answer before the command. Export validates the real tokenizer render
against an exact canary prefix and records the rendered-prefix SHA-256. The
contract sets `stop_on_newline=false` and records both `<|endoftext|>` and
`<|im_end|>` token IDs, along with the exact template and system-prompt hashes.
Those IDs are pinned as `151643/151645` for Compact and `248044/248046` for
both Qwen3.5 tiers; export refuses tokenizer drift.

The training system message ends with exactly one newline. That byte is part of
policy `howto.training.command-generation.v1`; its raw UTF-8 SHA-256 is
`66b3e1858ab9bf993f0a232ceb079fc256ff41e8177cfd3e949021e55c3a89e1`.
The manifest records both that raw-message hash and the family-specific
rendered canary hash. This distinction matters because the Qwen3.5 template
trims message content while Compact's ChatML renderer preserves it.

Production quality is a separate contract. Every exported manifest supplies
the exact non-retry policies currently deployed by HowTo:

- `howto.command-generation.macos.non-retry.v1`
- `howto.command-generation.linux.non-retry.v1`

For each target it records the system-message SHA-256 and the model-family
rendered canary SHA-256. A policy edit, platform wording edit, newline change,
or rendering change therefore invalidates prior quality evidence.

## Export on the GPU host

```bash
uv sync --extra export
make test-export
```

Run metadata-only preflight or LoRA export first:

```bash
uv run --extra export python -m howto.export \
  --model balanced \
  --export-mode lora \
  --base-revision "$BASE_REVISION" \
  --adapter-revision "$ADAPTER_REVISION" \
  --llama-cpp-dir "$LLAMA_CPP_DIR" \
  --llama-cpp-revision "$LLAMA_CPP_REVISION" \
  --preflight-only

make export-model-lora MODEL=balanced \
  BASE_REVISION="$BASE_REVISION" \
  ADAPTER_REVISION="$ADAPTER_REVISION" \
  LLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  LLAMA_CPP_REVISION="$LLAMA_CPP_REVISION" \
  ALLOW_DOWNLOADS=0
```

After the LoRA lane is proven, export a merged sweep for each tier:

```bash
make export-compact  BASE_REVISION="$COMPACT_BASE_REVISION"  LLAMA_CPP_DIR="$LLAMA_CPP_DIR" LLAMA_CPP_REVISION="$LLAMA_CPP_REVISION"
make export-balanced BASE_REVISION="$BALANCED_BASE_REVISION" LLAMA_CPP_DIR="$LLAMA_CPP_DIR" LLAMA_CPP_REVISION="$LLAMA_CPP_REVISION"
make export-accurate BASE_REVISION="$ACCURATE_BASE_REVISION" LLAMA_CPP_DIR="$LLAMA_CPP_DIR" LLAMA_CPP_REVISION="$LLAMA_CPP_REVISION"
```

Output directories are never overwritten. Each immutable manifest records
model/tier/display IDs, exact base and adapter revisions, converter hashes,
quantization, actual file bytes and SHA-256, reference BF16 metrics and their
report hashes, RAM estimate, template/prompt/stop provenance, and explicit
production qualification gaps.

## Runtime qualification and catalog

Evaluate the 400 held-out tasks through the actual Rust engine for every merged
quantization and for BF16 LoRA over pinned Q4/Q5/Q6 base GGUFs. Measure cold
start, warm latency, peak RSS, and command agreement on Apple Silicon and Linux
x86_64/ARM64. Do not assume quantization or runtime LoRA preserves BF16 quality.

For each selected merged Q4 artifact, create `qualification.json` beside the
manifest, artifact, and its referenced reports. `manifest_path`, every
`report_path`, and the artifact filename must be non-symlink sibling files.
`artifact_url` must be immutable, for example
`https://downloads.example/<artifact-sha256>/howto-qwen35-0.8b-q4_k_m.gguf`.

The qualification's common release fields are:

```json
{
  "model_id": "qwen35-0.8b",
  "manifest_path": "manifest.json",
  "manifest_sha256": "<sha256>",
  "artifact_url": "<credential-free immutable HTTPS URL>",
  "llama_cpp_revision": "<converter/export full commit>",
  "runtime_llama_cpp_revision": "5f55650a78f92aff4d48d671423e888fac0469ff",
  "llama_cpp_qualified": true,
  "training_base_revision_attestation": {
    "base_revision": "<same-full-base-commit>",
    "approved": true,
    "evidence": "<preserved training-cache record or retraining run>"
  },
  "distribution_rights": {
    "approved": true,
    "evidence": "<legal approval or approved open-teacher rebuild>"
  },
  "production_artifact_metrics": {
    "executable_pass_at_1": 0.66,
    "mean_latency_ms": 85.0,
    "peak_rss_bytes": 1300000000,
    "platform": "macos+linux arm64+x86_64 warm end-to-end worst-case aggregate"
  }
}
```

It also contains exactly four `production_quality_evaluations`, ordered
`macos-arm64`, `macos-x86_64`, `linux-arm64`, `linux-x86_64`. Each input record
has this exact evidence shape (values come from the manifest contract):

```json
{
  "target": "macos-arm64",
  "policy_id": "howto.command-generation.macos.non-retry.v1",
  "system_prompt_sha256": "<exact raw system prompt hash>",
  "rendered_prompt_sha256": "<family-specific rendered canary hash>",
  "runtime_llama_cpp_revision": "5f55650a78f92aff4d48d671423e888fac0469ff",
  "generation_policy_id": "howto.embedded.command-generation.deterministic.v1",
  "generation_policy_sha256": "<exact structured generation policy hash>",
  "executable_pass_at_1": 0.67,
  "task_count": 400,
  "task_corpus_sha256": "2457970a5bc708ffe642cf62ceb430e80c0493c78ab40d7caa4f71acd44fe204",
  "task_set_sha256": "<exact ordered held-out task-set hash>",
  "split_policy_id": "howto.group-kfold.fold-0.v1",
  "split_policy_sha256": "d23e9d22519f8d0aee184be49f179010e6db78d37889486f1a713e114367a684",
  "harness_revision": "<full immutable quality-harness commit>",
  "report_path": "quality-macos-arm64.json",
  "report_sha256": "<report sha256>"
}
```

The referenced quality report repeats those fields and adds
`schema_version: 1`, `evaluation_kind: "execution_grounded_pass_at_1"`, the
`model_id`, selected `artifact_sha256`, and `strict_retry: false`. All four
records must use the same task-set and quality-harness revisions, contain
exactly 400 tasks, and clear the tier's three-point quality floor.

The qualification also contains exactly four
`production_performance_evaluations` in the same target order:

```json
{
  "target": "macos-arm64",
  "runtime_llama_cpp_revision": "5f55650a78f92aff4d48d671423e888fac0469ff",
  "generation_policy_id": "howto.embedded.command-generation.deterministic.v1",
  "generation_policy_sha256": "<exact structured generation policy hash>",
  "mean_latency_ms": 75.0,
  "peak_rss_bytes": 1200000000,
  "platform": "macos-arm64 m2 warm end-to-end",
  "harness_revision": "<full immutable performance-harness commit>",
  "report_path": "performance-macos-arm64.json",
  "report_sha256": "<report sha256>"
}
```

Its report repeats those fields and adds `schema_version: 1`,
`evaluation_kind: "embedded_warm_end_to_end_performance"`, `model_id`, and
`artifact_sha256`. All four performance records must use the same performance
harness revision. The composer hashes and parses every report rather than
trusting copied metrics. The compatibility `production_artifact_metrics`
object is the worst-case aggregate across all four native targets.

Then compose the exact catalog consumed by setup:

```bash
make model-catalog \
  QUALIFICATIONS="dist/compact/qualification.json dist/balanced/qualification.json dist/accurate/qualification.json"
```

The composer re-hashes each manifest and model file, requires HTTPS, rejects a
credential, query, fragment, traversal, mutable branch/latest URL, or URL that
does not contain a known immutable revision/hash. It rejects a LoRA without its
base and unmerged converter sources, separately pins HowTo's embedded runtime
llama.cpp revision, enforces a 3-point maximum absolute pass@1 regression from
BF16 under both macOS and Linux prompt policies on all four native targets,
requires four per-target warm
end-to-end performance records, requires training-revision and
distribution-rights evidence, and emits
`model-catalog.json.sha256`. Only this qualified catalog marks model choices as
installable.

## Legal release gate

The published corpus used closed-model teacher outputs whose standard terms may
restrict training a competing model. Obtain written rights or rebuild with
approved open-weight teachers before distributing weights. Also record the
base-model licenses and publish completed model cards. A successful conversion
does not establish distribution rights.
