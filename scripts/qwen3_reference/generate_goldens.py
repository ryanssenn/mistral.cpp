#!/usr/bin/env python3
"""Generate per-module input/output goldens for qmog.cpp (layer 0, pos 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as scripts/qwen3_reference/generate_goldens.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_reference.harness import (  # noqa: E402
    Reference,
    assert_allclose,
    hf_forward,
    load,
    transformers_forward,
)
from qwen3_reference.goldens_to_cpp import write_inc  # noqa: E402
from qwen3_reference.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402

from qwen3_reference.harness import DEFAULT_HF_MODEL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT.parent / "Qwen3-0.6B-reference/checkpoint/config.json"
DEFAULT_WEIGHTS = REPO_ROOT.parent / "Qwen3-0.6B/model.safetensors"
DEFAULT_OUT = REPO_ROOT / "test/qwen/goldens"


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def stack_layers(num_layers: int) -> tuple[int, int, int]:
    """First, middle, and last decoder layer indices."""
    return 0, num_layers // 2, num_layers - 1


def collect_stack_goldens(ref: Reference, token_id: int, pos: int) -> dict[str, torch.Tensor]:
    """Hidden state after first, middle, and last decoder layers (pos 0 prefill)."""
    n = ref.config.num_hidden_layers
    first, middle, last = stack_layers(n)
    checkpoints = {first, middle, last}

    hidden = ref.embedding(token_id)
    entries: dict[str, torch.Tensor] = {
        "stack_token": torch.tensor([float(token_id)]),
        "stack_pos": torch.tensor([float(pos)]),
        "stack_num_layers": torch.tensor([float(n)]),
        "hidden_embedding": _flat(hidden),
    }

    for layer_idx in range(n):
        hidden = ref.layer(layer_idx, hidden, pos)
        if layer_idx in checkpoints:
            entries[f"hidden_L{layer_idx}_out"] = _flat(hidden)

    return entries


def collect_layer_goldens(ref: Reference, layer: int, token_id: int, pos: int) -> dict[str, torch.Tensor]:
    block = ref.layer_module(layer)
    attn = block.self_attn
    prefix = f"L{layer}"

    emb = ref.embedding(token_id)
    in_norm_out = ref.rms_norm(block.input_layernorm, emb)

    # q_norm / k_norm / rope intermediates (replay Qwen3Attention.forward lines)
    hidden = in_norm_out
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    q_proj = attn.q_proj(hidden).view(hidden_shape)
    k_proj = attn.k_proj(hidden).view(hidden_shape)
    q_norm_out = attn.q_norm(q_proj)
    k_norm_out = attn.k_norm(k_proj)

    query_states = q_norm_out.transpose(1, 2)
    key_states = k_norm_out.transpose(1, 2)
    cos, sin = ref.rotary_emb(hidden, pos)
    q_rope, k_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_out = ref.attention(layer, in_norm_out, pos)
    mid = emb + attn_out
    post_norm_out = ref.rms_norm(block.post_attention_layernorm, mid)
    mlp_out = ref.mlp(layer, post_norm_out)
    layer_out = ref.layer(layer, emb, pos)

    return {
        "embedding_token": torch.tensor([float(token_id)]),
        "embedding_out": _flat(emb),
        f"rmsnorm_{prefix}_input_in": _flat(emb),
        f"rmsnorm_{prefix}_input_out": _flat(in_norm_out),
        f"rmsnorm_{prefix}_q_in": _flat(q_proj),
        f"rmsnorm_{prefix}_q_out": _flat(q_norm_out),
        f"rmsnorm_{prefix}_k_in": _flat(k_proj),
        f"rmsnorm_{prefix}_k_out": _flat(k_norm_out),
        f"rope_{prefix}_q_in": _flat(query_states),
        f"rope_{prefix}_k_in": _flat(key_states),
        f"rope_{prefix}_cos": _flat(cos),
        f"rope_{prefix}_sin": _flat(sin),
        f"rope_{prefix}_q_out": _flat(q_rope),
        f"rope_{prefix}_k_out": _flat(k_rope),
        f"attn_{prefix}_in": _flat(in_norm_out),
        f"attn_{prefix}_out": _flat(attn_out),
        f"rmsnorm_{prefix}_post_in": _flat(mid),
        f"rmsnorm_{prefix}_post_out": _flat(post_norm_out),
        f"mlp_{prefix}_in": _flat(post_norm_out),
        f"mlp_{prefix}_out": _flat(mlp_out),
        f"layer_{prefix}_in": _flat(emb),
        f"layer_{prefix}_out": _flat(layer_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token", type=int, default=151643)
    parser.add_argument("--pos", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL if DEFAULT_HF_MODEL.exists() else None)
    parser.add_argument("--skip-hf-check", action="store_true")
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    ref = load(args.config, args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    module_entries = collect_layer_goldens(ref, args.layer, args.token, args.pos)
    inc_path = args.out_dir / "L0.inl"
    write_inc(inc_path, module_entries)

    stack_entries = collect_stack_goldens(ref, args.token, args.pos)
    stack_path = args.out_dir / "stack.inl"
    write_inc(stack_path, stack_entries, namespace="stack_golden_pos0")

    first, middle, last = stack_layers(ref.config.num_hidden_layers)

    modular_logits = ref.forward_token(args.token, args.pos)
    ref_logits = transformers_forward(ref, args.token, args.pos)
    max_err = assert_allclose(modular_logits, ref_logits, "vendored forward", atol=args.atol)

    hf_err = None
    if not args.skip_hf_check:
        if args.hf_model is None or not args.hf_model.exists():
            raise SystemExit("HF model not found; pass --hf-model or use --skip-hf-check")
        hf_logits = hf_forward(args.hf_model, args.token, args.pos)
        hf_err = assert_allclose(ref_logits, hf_logits, "HF transformers", atol=args.atol)

    print(f"wrote {len(module_entries)} keys -> {inc_path}")
    print(f"wrote {len(stack_entries)} keys -> {stack_path}  (layers {first}, {middle}, {last})")
    print(f"vendored forward match ok  max_err={max_err:.2e}")
    if hf_err is not None:
        print(f"HF transformers match ok  max_err={hf_err:.2e}")


if __name__ == "__main__":
    main()
