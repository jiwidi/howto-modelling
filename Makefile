.DEFAULT_GOAL := help
UV := uv run
PY := $(UV) python -m

# --- knobs ------------------------------------------------------------------
N          ?= 50
SEED       ?= 0
CANDIDATES ?= qwen3-0.6b,qwen25-coder-0.5b
FOLDS      ?= 0
N_SPLITS   ?= 5
EPOCHS     ?= 3
WORKERS    ?= 6
FOLD       ?= 0
N_PROMPTS  ?= 20
STEPS      ?= 32
BLOCK_SIZE ?= 32

# Everything trainable on the 5090: AR candidates + the block-diffusion track.
ALL_CANDIDATES ?= qwen3-0.6b,qwen25-coder-0.5b,qwen3-1.7b,qwen35-0.8b-base,qwen35-0.8b,qwen35-2b-base,qwen35-2b,gemma4-e2b,gemma3-1b,qwen3-0.6b-bd3lm

# Judge/teacher panel. `make generate-dataset` uses both by default; restrict
# with OPENAI=0 or ANTHROPIC=0.
OPENAI    ?= 1
ANTHROPIC ?= 1
PROVIDER_FLAGS := $(if $(filter 1,$(OPENAI)),--openai,) $(if $(filter 1,$(ANTHROPIC)),--anthropic,)

.PHONY: help setup check generate-dataset train evaluate cv all clean clean-runs

help:
	@echo "howto -- execution-grounded NL->shell distillation (PoC)"
	@echo ""
	@echo "  make setup             install deps with uv (python 3.12, torch cu128)"
	@echo "  make check             verify sandbox isolation + API keys + GPU"
	@echo "  make generate-dataset  N=$(N) teachers+judges: gpt-5.6-luna, claude-haiku-4.5"
	@echo "  make train             LoRA SFT for CANDIDATES on FOLDS"
	@echo "  make evaluate          executable pass@1 on held-out template families"
	@echo "  make cv                train+evaluate every fold (full cross-validation)"
	@echo "  make all               generate-dataset -> train -> evaluate"
	@echo ""
	@echo "Knobs: N, SEED, CANDIDATES, FOLDS, N_SPLITS, EPOCHS, OPENAI=0/1, ANTHROPIC=0/1"
	@echo "Candidates: qwen3-0.6b qwen25-coder-0.5b qwen3-1.7b qwen25-coder-1.5b"
	@echo ""
	@echo "Examples:"
	@echo "  make generate-dataset N=50"
	@echo "  make generate-dataset N=200 ANTHROPIC=0     # openai teacher only"
	@echo "  make cv CANDIDATES=qwen3-0.6b N_SPLITS=5"

setup:
	uv sync --python 3.12

check:
	@$(UV) python -c "\
import torch, howto.config as c, os; \
from howto import sandbox; \
print('GPU           :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); \
print('torch         :', torch.__version__); \
print('sandbox (bwrap):', 'ISOLATED' if sandbox.selftest() else 'FAILED - refusing to run'); \
print('openai key    :', 'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING'); \
print('anthropic key :', 'set' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING')"

generate-dataset:
	$(PY) howto.generate_dataset $(N) $(PROVIDER_FLAGS) --seed $(SEED) --workers $(WORKERS)

train:
	$(PY) howto.train --candidates $(CANDIDATES) --folds $(FOLDS) \
		--n-splits $(N_SPLITS) --epochs $(EPOCHS)

evaluate:
	$(PY) howto.evaluate --candidates $(CANDIDATES) --folds $(FOLDS) --n-splits $(N_SPLITS) \
		--steps $(STEPS) --block-size $(BLOCK_SIZE)

benchmark:
	$(PY) howto.benchmark --candidates $(CANDIDATES) --fold $(FOLD) \
		--n-prompts $(N_PROMPTS) --steps $(STEPS) --block-size $(BLOCK_SIZE)

# The full tournament: every autoregressive candidate plus the diffusion track.
tournament:
	$(MAKE) train     CANDIDATES="$(ALL_CANDIDATES)"
	$(MAKE) evaluate  CANDIDATES="$(ALL_CANDIDATES)"
	$(MAKE) benchmark CANDIDATES="$(ALL_CANDIDATES)"

cv:
	$(MAKE) train FOLDS=all
	$(MAKE) evaluate FOLDS=all

all: generate-dataset train evaluate

clean-runs:
	rm -rf runs/* reports/*

clean: clean-runs
	rm -rf data/raw/* data/processed/*
