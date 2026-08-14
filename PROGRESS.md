# Progress — execution-grounded NL→shell distillation

Session 2, 2026-08-12. Status: **full pipeline works, including the diffusion
track.** 10 models train and evaluate; latency benchmark exists and shows the
parallel-decoder speedup clearly. Accuracy numbers are still statistically
underpowered — full cross-validation was running at end of session.

---

## Commands

```bash
make check                 # GPU + sandbox isolation + API keys
make generate-dataset      # 50 tasks, gpt-5.6-luna + claude-haiku-4.5
make train                 # LoRA SFT (autoregressive or diffusion, auto-routed)
make evaluate              # executable pass@1 on held-out template families
make benchmark             # response time, tok/s, peak VRAM
make tournament            # all 10 candidates: train + evaluate + benchmark
make cv                    # train + evaluate every fold

# quality-vs-latency sweep for a block-diffusion candidate
uv run python -m howto.sweep_diffusion --candidate qwen3-0.6b-bd3lm --fold 0
```

---

## Session 3 (2026-08-13): 10k corpus, all 10 models trained + evaluated

Corpus: 10,000 tasks / 20,000 SFT records / 1,768 distinct signatures, built in
15.8 min at 55.6% yield. **Gemini-only** (`--author gemini`) -- the even
three-way round-robin was built but never ran, so students learn one family's
shell idioms.

Results at n=400 held-out tasks per model, template-disjoint, temperature 0.
Safety clean everywhere: 0.00% forbidden side-effects, unintended writes <=1.5%.

| model | base | SFT | delta | 1-line | mean ms | peak GB |
|---|--:|--:|--:|--:|--:|--:|
| qwen35-2b | 27.0% | 74.5% | +47.5 | **3%** | 857 | 3.96 |
| qwen35-2b-base | 24.2% | 71.8% | +47.5 | 60% | 437 | 3.96 |
| **qwen35-0.8b** | 14.2% | **67.8%** | +53.5 | 98% | 261 | 1.62 |
| qwen35-0.8b-base | 8.5% | 67.5% | +59.0 | 57% | 404 | 1.62 |
| qwen3-1.7b | 22.8% | 67.0% | +44.2 | 100% | 116 | 4.38 |
| gemma4-e2b | 38.8% | 65.5% | +26.8 | 100% | 250 | 10.61 |
| qwen25-coder-0.5b | 15.8% | 63.7% | +48.0 | 100% | 84 | 1.14 |
| qwen3-0.6b | 4.0% | 59.8% | +55.8 | 100% | 113 | 1.67 |
| gemma3-1b | 7.5% | 50.2% | +42.7 | 100% | 161 | 2.22 |
| qwen3-0.6b-bd3lm | 0.0% | 1.5% | +1.5 | 100% | 99 | 1.36 |

### BLOCKER: Qwen3.5 students never learn to emit EOS

`qwen35-2b` scores highest but is not shippable. It emits the correct command,
then hallucinates the next conversation turn:

```
sed -n '2~2p' server.conf
user
Count the total number of lines across all files...
assistant
<think>

</think>

grep -h 'enabled' config/*.conf | wc -l
```

One bug, two symptoms that looked unrelated:
* 3% one-line compliance (pass@1 survives only because `clean_command` takes
  line 1)
* 857 ms mean latency, slowest in the fleet -- it always runs to the 96-token
  cap instead of stopping at ~15 tokens

Also hits `qwen35-2b-base` (60%) and `qwen35-0.8b-base` (57%), i.e. the Qwen3.5
family specifically. Fix in the SFT data path: the assistant turn needs a proper
EOS for the Qwen3.5 chat template. **Until then treat 74.5% as unshippable and
prefer `qwen35-0.8b` (67.8% @ 98% format) or `qwen25-coder-0.5b`
(63.7% @ 84 ms, 1.14 GB).**

### Other findings

* Verified data works: every AR model gained 27-59 points, vs +16 for the same
  recipe on the old 88-row corpus. Confounds verification with the 200x data
  increase, so it is not a clean H1 test.
* `qwen35-0.8b` (67.8%) beats `qwen3-1.7b` (67.0%) at half the size -- supports
  the nano-track hypothesis, though the gap is inside noise at n=400.
* Diffusion track has failed on quality: 1.5% pass@1, 48.8% syntax validity
  (vs ~100% AR). 20x more data did not help and the earlier sweep ruled out
  decoding knobs. Speed edge also eroded: 24.9 ms untuned -> 99.4 ms tuned.

### Infrastructure added

* `scripts/pipeline.sh` -- detached supervisor (train -> evaluate -> benchmark)
  with a STATUS heartbeat; launch under `setsid` so it survives the terminal.
* `scripts/train_all.sh` -- VRAM-grouped parallel waves + per-model retry.
* Parallelism: batch 4x accum4 -> 16x accum1 (GPU 58% -> 100%), verification
  thread pool (60 tasks: 6 s -> 0.3 s), generation batch 8 -> 32.
* `evaluate --max-test N` for subsampled folds.

**Operational trap worth remembering:** `pgrep -f "howto.train"` matches the
watcher's own command line, so any loop waiting on it waits forever. This
deadlocked two runs. Wait on real PIDs (`kill -0 $PID`) or use a bracket pattern
(`[h]owto.train`). Equally: an adapter directory is created at trainer init, so
its existence proves nothing -- `run_meta.json` is the completion marker.

---

## Resolved in session 2

### 1. Gemma 4 zero-gradient bug — FIXED

Root cause: Gemma 4 is a multimodal checkpoint, and a bare `q_proj` suffix
target also matched **the vision tower** (`model.vision_tower.encoder...`),
whose `Gemma4ClippableLinear` PEFT cannot wrap. LoRA attached to modules that
never run in a text-only forward pass → `grad_norm: 0`, flat loss, useless
adapter that still saved successfully.

Fix: scope `lora_targets` to `.*language_model.*\.(q_proj|...)$`, where the
projections are plain `nn.Linear`. Now: loss 5.25 → 0.73, token accuracy
48% → 89%.

**Guard added:** `assert_gradients_flow()` runs one forward/backward before every
training run and hard-fails if zero adapter tensors receive a gradient. This
class of silent failure cannot recur.

### 2. Diffusion track — IMPLEMENTED

Both training and inference now work for block-diffusion models.

- `src/howto/diffusion.py` — BD3LM decoding: staircase attention mask (causal
  across blocks, bidirectional within), iterative low-confidence remasking,
  adapted from the dllm-hub model card reference (Apache-2.0).
- `src/howto/train_diffusion.py` — masked-denoising LoRA SFT. Masks a random
  fraction `t` of *response* tokens, cross-entropy on masked positions with the
  standard `1/t` MDLM importance weight, prompt never masked and never in the
  loss (matches `completion_only_loss=True` on the AR side so the tracks are
  comparable). Trained: loss 4.12 → 2.77.
- `src/howto/vendor/a2d_qwen3.py` — vendored the A2D modeling code. Two reasons:
  its `if __name__ == "__main__"` block imports `dllm` (not on PyPI), which
  transformers' *static* import scan rejects even though the block never runs;
  and transformers 5.x removed `DecoderLayer.attention_type`, which the original
  file dereferences. Vendoring also drops `trust_remote_code`.

### 3. Qwen3.5 + Gemma 3 added

All four Qwen3.5 models exist, are ungated, and ship chat templates:
`Qwen3.5-0.8B-Base`, `Qwen3.5-0.8B`, `Qwen3.5-2B-Base`, `Qwen3.5-2B`.
Gemma 3 works now that `HF_TOKEN` is exported from `.env` (config.py normalises
it, along with `HUGGING_FACE_HUB_TOKEN`).

---

## Results

### Latency — the headline finding

20 prompts, single request, bf16, RTX 5090, `max_new_tokens=96`,
diffusion at `steps=32, block_size=32`:

| model | variant | decoder | mean ms | p95 ms | tok/s | peak GB |
|---|---|---|--:|--:|--:|--:|
| **qwen3-0.6b-bd3lm** | base | **block_diffusion** | **36.9** | 66.5 | **856.8** | 1.25 |
| **qwen3-0.6b-bd3lm** | sft | **block_diffusion** | **37.2** | 66.9 | **849.8** | 1.36 |
| qwen25-coder-0.5b | sft | autoregressive | 71.5 | 130.1 | 212.7 | 1.14 |
| qwen3-1.7b | base | autoregressive | 104.3 | 257.8 | 146.2 | 4.39 |
| gemma3-1b | base | autoregressive | 132.6 | 264.9 | 116.2 | 2.02 |
| qwen35-2b-base | base | autoregressive | 171.5 | 237.4 | 91.2 | 3.81 |
| gemma4-e2b | base | autoregressive | 184.7 | 449.1 | 81.2 | 10.24 |
| qwen35-2b | sft | autoregressive | 188.0 | 319.8 | 92.6 | 3.96 |

**Block diffusion is 2–6× faster than every autoregressive model** and ~4–9×
higher throughput. The speed hypothesis is confirmed: latency scales with
diffusion *steps*, not token count, which is exactly right for one-line
commands. Lowering `steps` below 32 should go faster still — untested.

**But it scored 0% pass@1.** Fast and wrong. See below.

### Accuracy — proper 5-fold cross-validation (n=50)

Only the two cheapest candidates finished all 5 folds before the run was
stopped. These are the trustworthy numbers:

| model | base pass@1 | SFT pass@1 | delta | 1-line format (base → SFT) |
|---|--:|--:|--:|--:|
| qwen25-coder-0.5b | 36% | **48%** | **+12** | 22% → 100% |
| qwen3-0.6b | 10% | **26%** | **+16** | 98% → 100% |

Both deltas are positive at n=50, which **reverses the fold-0 picture below** —
the alarming negative deltas there were noise, exactly as suspected. Verified
data helps at this scale.

The format column is a second real effect: untuned Qwen2.5-Coder-0.5B emits a
bare one-line command only 22% of the time (it wraps things in prose/markdown);
after SFT it is 100%. The deployment contract is learned reliably even where
correctness is still weak.

### Diffusion quality — the 0% is a model problem, not a knob problem

Sweep over (steps, block_size, max_new_tokens) on fold 0, SFT variant:

| steps | block | max_new | pass@1 | syntax ok | mean ms |
|--:|--:|--:|--:|--:|--:|
| 8 | 16 | 32 | 0% | 60% | 70.1 |
| 16 | 16 | 32 | 0% | 80% | 88.1 |
| **32** | **16** | **32** | **10%** | **100%** | 135.7 |
| 8 | 32 | 64 | 0% | 80% | 30.5 |
| 32 | 32 | 64 | 0% | 100% | 43.7 |
| 64 | 32 | 64 | 0% | 100% | 91.2 |

More steps buys *syntactic validity* (60% → 100%) but not correctness. Beyond
32 steps nothing improves, so the ceiling is not in the decoding schedule.

Failure mode, at the best configuration — right utilities, wrong structure:

```
got: sort -n 3 data/batch_access.txt | uniq -c
ref: sort data/batch_access.txt | uniq -c | sort -rn | head -n 3 | awk '{print $2}'

got: find inbox -name '*.conf' -exec {} \;
ref: find inbox -type f -name '*.conf'
```

It selects the correct tools but truncates pipelines, folds a later stage's
argument into an earlier command (`sort -n 3` for `... | head -n 3`), and emits
incomplete constructs. This is what parallel block denoising would be expected
to get wrong: tokens are committed without the left-to-right dependency that
makes pipeline composition coherent. Fold 0's held-out families are both
multi-utility pipelines, i.e. the hardest case for it.

For reference the *autoregressive* Qwen3-0.6B — same base model — reaches 26%
over 5 folds. So the AR→diffusion adaptation currently costs most of the task
quality at this scale.

### Accuracy — fold 0 only, NOT trustworthy (kept for contrast)

| model | base pass@1 | sft pass@1 | delta |
|---|--:|--:|--:|
| gemma4-e2b | 90% | **100%** | +10% |
| qwen35-2b | 80% | 0% | −80% |
| qwen35-2b-base | 70% | 50% | −20% |
| gemma3-1b | 40% | 0% | −40% |
| qwen35-0.8b-base | 30% | 0% | −30% |
| qwen25-coder-0.5b | 30% | 0% | −30% |
| qwen3-0.6b | 20% | 30% | +10% |
| qwen35-0.8b | 10% | 20% | +10% |
| qwen3-1.7b | 0% | 0% | 0% |
| qwen3-0.6b-bd3lm | 0% | 0% | 0% |

**Do not draw conclusions from this table** — kept only to show how misleading a
single fold is. Fold 0 holds out just **2 template families = 10 tasks**, both
multi-utility pipeline families, so a model either gets the pipeline shape right
or scores zero — hence the 0%/100% bimodality. One task is worth 10 points, and
the 5-fold numbers above flip the sign of every large negative delta that could
be rechecked.

Inspected failures are *genuine wrong answers*, not a scoring bug — e.g.
`find inbox -type f -name '*.conf' -print0 | xargs -0 basename -s .conf | sort -u`
where the task wanted full paths. The evaluator is behaving correctly.

The large negative deltas are plausible overfitting (88 training rows, 3 epochs,
held-out families), but at this n it is indistinguishable from noise.

---

## Trained adapters

`runs/`, 18 adapters:

- **All 5 folds** (cross-validated, trustworthy): `qwen3-0.6b`,
  `qwen25-coder-0.5b`.
- **Fold 0 only**: `qwen3-1.7b`, `qwen35-0.8b-base`, `qwen35-0.8b`,
  `qwen35-2b-base`, `qwen35-2b`, `gemma4-e2b`, `gemma3-1b`, `qwen3-0.6b-bd3lm`.

The all-folds run for the remaining 8 was stopped partway. Resume with:

```bash
make train CANDIDATES="qwen3-1.7b,qwen35-0.8b-base,qwen35-0.8b,qwen35-2b-base,qwen35-2b,gemma4-e2b,gemma3-1b,qwen3-0.6b-bd3lm" FOLDS=all
make evaluate CANDIDATES="<same>" FOLDS=all
```

Budget ~45 min on the 5090; model loading dominates, not training.

---

## Environment facts worth keeping

- torch 2.11.0+cu128, transformers 5.15.0, trl 1.9.2, python 3.12, uv, RTX 5090.
- **The OpenAI key is EU-region-scoped.** `api.openai.com` 401s with "outside
  project geography EU"; use `https://eu.api.openai.com/v1` (pinned in
  `teachers.py`, override via `OPENAI_BASE_URL`).
- `RLIMIT_NPROC` is per-UID machine-wide and counts **threads**; the fork-bomb
  cap must sit above the login's current thread count or bwrap fails with EAGAIN.
- TRL 1.9 `SFTConfig` has no `warmup_ratio` (use `warmup_steps`), and needs
  `processing_class=tok` for multimodal checkpoints or it builds a vision
  processor and demands `pillow`.
- `apply_chat_template(tokenize=True)` returns an `Encoding` in transformers 5.x,
  not a list of ints.

---

## TODO, in priority order

1. **Finish the 5-fold cross-validation for the remaining 8 models** (commands
   above). Two are done; the other eight still have single-fold numbers only.
2. **Close the diffusion quality gap.** The sweep rules out decoding knobs, so
   the remaining levers are training-side: (a) more masked-SFT epochs and a
   different `t` range — currently U(0.15, 0.95), 3 epochs on 88 rows;
   (b) far more task data (the binding constraint everywhere); (c) a
   semi-autoregressive decode with `block_size` ≈ 8 so pipeline stages commit
   more sequentially — worth testing since the errors are compositional;
   (d) accept the tradeoff and treat the diffusion model as a draft generator
   with an AR verifier. Note more steps did fix syntax validity, so the decoder
   itself is sound.
3. **Convert Qwen3.5-0.8B/2B to diffusion** (requested). Not started. The A2D
   recipe is in `vendor/a2d_qwen3.py` (an `A2DQwen3Config` wrapper over the stock
   Qwen3 model + block attention); the conversion needs an adaptation phase on
   general text before task SFT, which is not free. Reference:
   github.com/ZHZisZZ/dllm. Note Qwen3.5 is `model_type: qwen3_5`, so the
   vendored Qwen3 wrapper needs porting.
4. **Scale the dataset.** 50 tasks / 10 families is the binding constraint on
   every number above. Plan targets 80k–120k unique specs and a frozen
   2,000-task private benchmark. Also missing: BSD/GNU portability,
   ambiguity/abstention, adversarial-safety categories.
5. **Statistics**: 3 seeds, McNemar on paired outcomes, paired bootstrap CIs,
   Holm correction.
6. **Later recipe stages**: execution-ranked DPO (seed negatives from student
   failures — `reports/evaluation.json` logs them), on-policy GKD, RLVR.
7. **Deployment**: GGUF export, quantization sweep, C++ `libllama` / Rust
   runtimes, independent AST safety checker.

---

## Licensing caveat (still open)

Standard OpenAI/Anthropic terms restrict using outputs to train competing
models. This PoC uses both as teachers because that was explicitly requested for
validation. Before any published training run, get a written exception or swap in
the self-hosted open-weight committee — `PROVIDERS` in `config.py` is the only
place that changes.
