# howto — execution-grounded NL→shell distillation

Small local models that turn a natural-language request into a shell command,
trained on a corpus where **a sandbox — not a model — decides what is correct**.

Instead of collecting natural-language/command pairs, we generate *executable
task specifications*: a workspace, a request, and postconditions. Candidate
commands are executed in an isolated sandbox with complete before/after
filesystem snapshots, and only what actually satisfies the postconditions
becomes training data.

```
task spec  →  reference self-validation  →  multi-model proposals
   →  static filter  →  sandboxed execution (full side-effect snapshot)
   →  randomized replay  →  verified positives
   →  LoRA SFT on one GPU  →  executable pass@1 on held-out task shapes
```

## Results

10,000 verified tasks / 20,000 SFT records. Evaluated on **400 held-out tasks**
per model, grouped so no task *shape* is shared between train and test,
temperature 0. LoRA rank 32, 1 epoch, single RTX 5090.

| model | base | **SFT** | delta | 1-line | mean ms | peak GB |
|---|--:|--:|--:|--:|--:|--:|
| **qwen35-2b** | 27.0% | **74.5%** | +47.5 | 100% | 182 | 3.96 |
| qwen35-2b-base | 24.2% | 71.8% | +47.5 | 100% | 187 | 3.96 |
| qwen35-0.8b | 14.2% | 67.8% | +53.5 | 100% | 180 | 1.62 |
| qwen35-0.8b-base | 8.5% | 67.5% | **+59.0** | 100% | 178 | 1.62 |
| qwen3-1.7b | 22.8% | 67.0% | +44.2 | 100% | 118 | 4.38 |
| gemma4-e2b | 38.5% | 65.5% | +27.0 | 99% | 277 | 10.61 |
| **qwen25-coder-0.5b** | 15.8% | 63.7% | +48.0 | 100% | **86** | **1.14** |
| qwen3-0.6b | 4.0% | 59.8% | +55.8 | 100% | 116 | 1.67 |
| gemma3-1b | 7.5% | 50.2% | +42.7 | 100% | 165 | 2.22 |
| qwen3-0.6b-bd3lm *(diffusion)* | 0.0% | 1.5% | +1.5 | 100% | 101 | 1.36 |

*pass@1 = the command actually produced the required result in a sandbox.
1-line = emitted exactly one bare command, the deployment contract.
Latency is a single warm request, bf16.*

**Every autoregressive model gained 27–59 points.** Safety was clean throughout:
**0.00% forbidden side-effects** across all 20 model-variants, unintended writes
≤1.5%.

Two candidates worth shipping: **`qwen35-2b`** for accuracy (74.5%, 182 ms,
4 GB), **`qwen25-coder-0.5b`** for footprint (63.7%, **86 ms, 1.14 GB**) — 11
points cheaper for 2× the speed and a third of the memory.

The **block-diffusion track failed on quality** (1.5% pass@1, 48.8% syntax
validity). It decodes a whole block in parallel and is genuinely fast, but at
0.6B it cannot hold pipeline structure together — it picks the right utilities
and then mangles the composition.

## Why the numbers are trustworthy

**The oracle is the postcondition, not a reference string.** Many textually
unrelated commands are correct. Each task carries ≥2 independent reference
solutions and is *rejected* unless both satisfy the oracle. This caught a real
generator bug on the first run (`du … | xargs basename` silently returning a
byte count).

**Correctness uses complete filesystem snapshots** — content hash, mode and
symlink target for every path — not stdout alone. A command that prints the
right answer while corrupting the workspace is scored as a failure.

**Splits are shape-disjoint, not random.** Random row splits leak the command
shape into the test fold. `GroupKFold` groups by a structural signature
(utility set + pipeline depth + file count), giving 1,765 groups; the split
asserts at runtime that no group spans train and test.

**LLM-authored tasks are never trusted about behaviour.** The model invents the
workspace, request and solutions; the sandbox then *executes* the first solution
to derive the postconditions and requires an independent second solution to
reproduce them byte-for-byte. Roughly 22% of authored tasks are discarded
because the two "equivalent" commands disagree — silent label noise in a
conventional synthetic pipeline.

Guarantee ladder, weakest to strongest:

1. model claims a command works — *never trusted alone*
2. command executes and we record everything it did — *derived oracle*
3. a second, textually different command reproduces it — *accepted here*
4. re-instantiation with randomized fixtures — *procedural tasks only*

## Quick start

```bash
make setup              # uv sync: python 3.12, torch cu128 (Blackwell)
make check              # GPU + sandbox isolation + API keys
make generate-dataset   # small verified dataset via the teacher committee
make train              # LoRA SFT
make evaluate           # executable pass@1 on held-out shapes
make benchmark          # latency, tok/s, peak VRAM
```

Build a large corpus (Gemini/OpenAI/Anthropic round-robin, evenly balanced on
*accepted* tasks so no single family dominates):

```bash
uv run python -m howto.build_corpus 10000 --author gemini,openai,anthropic --workers 32
bash scripts/pipeline.sh     # train → evaluate → benchmark, unattended
```

## Sandbox

Model-generated shell runs under `bubblewrap` with `--unshare-all` (no network),
read-only system paths, a fresh disposable workspace per run, `LC_ALL=C`/`TZ=UTC`,
plus wall-clock, file-size and fork-bomb limits. `make check` asserts all three
properties before anything executes: commands run, the network is unreachable,
the host filesystem is read-only.

## Layout

| path | role |
|---|---|
| `src/howto/spec.py` | task spec, execution result, verdict schemas |
| `src/howto/tasks.py` | procedural generators (10 template families) |
| `src/howto/authoring.py` | LLM-authored tasks with execution-derived oracles |
| `src/howto/sandbox.py` | bubblewrap isolation + filesystem snapshot/diff |
| `src/howto/verify.py` | postconditions, replay, task self-validation |
| `src/howto/teachers.py` | OpenAI / Anthropic / Gemini proposers and judges |
| `src/howto/build_corpus.py` | large-corpus builder with dedup and checkpointing |
| `src/howto/splits.py` | leakage-resistant grouped cross-validation |
| `src/howto/train.py` | LoRA SFT |
| `src/howto/train_diffusion.py` | masked-denoising SFT for block-diffusion students |
| `src/howto/diffusion.py` | BD3LM parallel decoding |
| `src/howto/evaluate.py` | executable pass@1, safety, format metrics |
| `src/howto/benchmark.py` | response time, throughput, memory |
| `runs/` | trained LoRA adapters (git-lfs) |
| `reports/` | evaluation, latency, summary tables |

## Known issues and limits

- **Fold 0 only.** No confidence intervals; the 67.8 / 67.5 / 67.0 cluster is
  inside noise at n=400. Full CV is `FOLDS=all`.
- **This corpus is Gemini-only.** The three-way round-robin exists but was not
  used for the published run, so students learn one family's shell idioms.
- **Qwen3.5 needs an explicit stop token.** Its `generation_config.eos_token_id`
  is `<|endoftext|>` while its chat template ends assistant turns with
  `<|im_end|>`, so `generate()` runs past the token the model was trained to
  emit and hallucinates further turns. `stop_token_ids()` in `evaluate.py`
  handles this; anything consuming these adapters must do the same.
- Later recipe stages (execution-ranked DPO, on-policy GKD, RLVR) are not built.
- No GGUF export or quantization sweep yet.

## Licensing note

Standard OpenAI and Anthropic terms restrict using model outputs to train
competing models. The teacher committee is used here for research validation;
obtain a written exception or swap in self-hosted open-weight teachers before
any production training run — `PROVIDERS` in `config.py` is the only place that
changes.
