"""Block-diffusion (BD3LM) decoding for the parallel-decode track.

Adapted from the reference implementation on the dllm-hub model cards
(https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1, Apache-2.0),
which accompanies "dLLM: Simple Diffusion Language Modeling" and BD3LM
(arXiv:2503.09573).

Unlike autoregressive decoding, a whole block of `block_size` tokens is denoised
together: every step the model sees the partially-unmasked block, predicts all
masked positions at once, and commits the most confident subset. Latency is
therefore governed by the number of diffusion *steps*, not by the number of
tokens -- which is exactly why it is interesting for one-line shell commands.
Set `steps < max_new_tokens` to trade quality for speed.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

DEFAULT_BLOCK_SIZE = 32


def load_diffusion_model(hf_id: str, adapter: str | None = None, device: str = "cuda:0"):
    """Diffusion students load through AutoModelForMaskedLM, not CausalLM."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # Importing the vendored module registers a2d-qwen3 with the auto classes,
    # so no trust_remote_code and no phantom `dllm` dependency.
    from .vendor import a2d_qwen3  # noqa: F401

    tok = AutoTokenizer.from_pretrained(adapter or hf_id)
    model = AutoModelForMaskedLM.from_pretrained(
        hf_id, dtype=torch.bfloat16
    ).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def resolve_mask_id(tok) -> int:
    mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        raise SystemExit(
            "tokenizer exposes no mask_token_id; a masked-diffusion decoder "
            "cannot run without one"
        )
    return mask_id


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    g = (-torch.log(noise)) ** temperature
    return logits.exp() / g


def _num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """How many masked positions to commit at each step (evenly spread)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    out = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.long
    ) + base
    for i in range(mask_num.size(0)):
        out[i, : rem[i]] += 1
    return out


def _staircase_attention_mask(x: torch.Tensor, block_size: int, pad_id: int):
    """Causal across blocks, bidirectional within a block."""
    B, T = x.shape
    device = x.device

    valid = x != pad_id
    pos_raw = torch.cumsum(valid.long(), dim=-1)
    position_ids = torch.where(valid, pos_raw - 1, torch.zeros_like(pos_raw)).long()

    col = torch.arange(T, device=device)
    block_ids = (col // block_size).view(1, T).expand(B, T)
    block_ids = torch.where(valid, block_ids, torch.full_like(block_ids, -1))

    q = block_ids.view(B, 1, T, 1)
    k = block_ids.view(B, 1, 1, T)
    attn = (k <= q) & (q >= 0) & (k >= 0)
    return attn, position_ids


def _commit_step(
    logits: torch.Tensor,
    x_block: torch.Tensor,
    mask_block: torch.Tensor,
    num_transfer: torch.Tensor,
    temperature: float,
    remasking: str,
) -> torch.Tensor:
    B, L, _ = logits.shape
    if not mask_block.any():
        return x_block

    noisy = _add_gumbel_noise(logits, temperature)
    x0 = noisy.argmax(dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        conf = torch.rand((B, L), device=logits.device)
    else:
        raise ValueError(f"unknown remasking strategy {remasking!r}")

    x0 = torch.where(mask_block, x0, x_block)
    conf = torch.where(mask_block, conf, torch.full_like(conf, -float("inf")))

    commit = torch.zeros_like(x_block, dtype=torch.bool)
    for i in range(B):
        k = int(num_transfer[i].item())
        if k > 0:
            available = int((conf[i] > -float("inf")).sum().item())
            k = min(k, available)
            if k > 0:
                _, idx = torch.topk(conf[i], k)
                commit[i, idx] = True

    out = x_block.clone()
    out[commit] = x0[commit]
    return out


@torch.no_grad()
def diffusion_generate(
    model,
    tok,
    input_ids: torch.Tensor,
    steps: int = 32,
    max_new_tokens: int = 64,
    block_size: int = DEFAULT_BLOCK_SIZE,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """Return the full sequence (prompt + generated block(s))."""
    device = model.device
    mask_id = resolve_mask_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    x = input_ids.to(device).long()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B = x.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    num_blocks = max(math.ceil(max_new_tokens / block_size), 1)
    steps_per_block = max(math.ceil(steps / num_blocks), 1)
    generated = 0

    while generated < max_new_tokens and not finished.all():
        t_prefix = x.size(1)
        offset = t_prefix % block_size
        room = block_size if offset == 0 else block_size - offset
        cur_len = min(room, max_new_tokens - generated)
        if cur_len <= 0:
            break

        attn_pfx, pos_pfx = _staircase_attention_mask(x, block_size, pad_id)
        cond_past = model(
            x, attention_mask=attn_pfx, position_ids=pos_pfx, use_cache=True
        ).past_key_values

        block = torch.full((B, cur_len), mask_id, device=device, dtype=torch.long)
        block[finished] = pad_id
        x = torch.cat([x, block], dim=1)
        t_total = x.size(1)

        block_mask = x[:, -cur_len:] == mask_id
        num_transfer = _num_transfer_tokens(block_mask, steps_per_block)

        full_attn, full_pos = _staircase_attention_mask(x, block_size, pad_id)
        attn_blk = full_attn[:, :, t_prefix:t_total, :]
        pos_blk = full_pos[:, t_prefix:t_total]

        for t in range(num_transfer.size(1)):
            x_blk = x[:, t_prefix:t_total]
            m_blk = x_blk == mask_id
            if not m_blk.any():
                break
            logits = model(
                x_blk,
                attention_mask=attn_blk,
                position_ids=pos_blk,
                past_key_values=copy.deepcopy(cond_past),
                use_cache=False,
            ).logits
            x[:, t_prefix:t_total] = _commit_step(
                logits, x_blk, m_blk, num_transfer[:, t], temperature, remasking
            )
            if tok.eos_token_id is not None:
                finished |= (x[:, t_prefix:t_total] == tok.eos_token_id).any(dim=1)
            if finished.all():
                break

        generated += cur_len

    return x


def decode_new(tok, full: torch.Tensor, prompt_len: int) -> list[str]:
    """Strip the prompt and any mask/pad residue from each generated row."""
    mask_id = resolve_mask_id(tok)
    outputs = []
    for row in full:
        new = row[prompt_len:]
        keep = new[new != mask_id]
        text = tok.decode(keep, skip_special_tokens=True)
        outputs.append(text)
    return outputs
