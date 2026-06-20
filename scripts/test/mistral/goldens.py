"""
Generate test/mistral/expected.txt and test/mistral/logits_expected.txt.

Usage (from repo root, with ../Mistral-7B-v0.1 present):
  python scripts/test/mistral/goldens.py

Set DUMP_LAYER_STACK=1 to include layer-stack keys in logits_expected.txt.
"""
import gc
import json
import math
import os

import safetensors
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL = os.environ.get(
    "MISTRAL_MODEL",
    os.path.abspath(os.path.join(REPO_ROOT, "../Mistral-7B-v0.1")),
)
EXPECTED_OUT = os.path.join(REPO_ROOT, "test/mistral/expected.txt")
LOGITS_OUT = os.path.join(REPO_ROOT, "test/mistral/logits_expected.txt")

PROMPTS = {
    "sky": "The color of the sky is",
    "paris": "Paris is the capital of",
}
NUM_GEN_STEPS = 5
TOPK = 10
DUMP_LAYER_STACK = os.environ.get("DUMP_LAYER_STACK", "0") == "1"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def dump_vector(f, name, tensor):
    if isinstance(tensor, torch.Tensor):
        values = tensor.detach().float().flatten().tolist()
    else:
        values = list(tensor)
    f.write(name + "\n")
    f.write(" ".join(str(float(v)) for v in values) + "\n")


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return {
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim,
        "rope_theta": cfg["rope_theta"],
        "norm_eps": cfg["rms_norm_eps"],
    }


def load_weight(model_dir, name):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    shard = os.path.join(model_dir, weight_map[name])
    with safetensors.safe_open(shard, framework="pt") as handle:
        return handle.get_tensor(name).float()


def compute_inv_freq(rope_theta, head_dim):
    n = head_dim // 2
    return [1.0 / (rope_theta ** (i / n)) for i in range(n)]


def compute_cos_sin(inv_freq, pos, head_dim):
    cos = [math.cos(inv_freq[i % len(inv_freq)] * pos) for i in range(head_dim)]
    sin = [math.sin(inv_freq[i % len(inv_freq)] * pos) for i in range(head_dim)]
    return cos, sin


def apply_mistral_rope(flat, cos, sin, n_heads=1, head_dim=None):
    if head_dim is None:
        head_dim = len(cos)
    half = head_dim // 2
    out = flat.clone()
    for h in range(n_heads):
        start = h * head_dim
        for i in range(half):
            xi = flat[start + i]
            yi = flat[start + i + half]
            c = cos[i]
            s = sin[i]
            out[start + i] = xi * c - yi * s
            out[start + i + half] = xi * s + yi * c
    return out


def rmsnorm(x, weight, eps):
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return x / rms * weight


def silu(x):
    return x / (1.0 + torch.exp(-x))


def dump_rotary_and_rope(f, cfg):
    inv_freq = compute_inv_freq(cfg["rope_theta"], cfg["head_dim"])
    dump_vector(f, "inv_freq", inv_freq)

    for pos, suffix in ((0, "0"), (3, "3")):
        cos, sin = compute_cos_sin(inv_freq, pos, cfg["head_dim"])
        dump_vector(f, f"cos{suffix}", cos)
        dump_vector(f, f"sin{suffix}", sin)

    head_dim = cfg["head_dim"]
    q = torch.tensor([(i % head_dim) / 256.0 for i in range(head_dim)], dtype=torch.float32)
    inv = compute_inv_freq(cfg["rope_theta"], head_dim)
    for pos in range(4):
        cos, sin = compute_cos_sin(inv, pos, head_dim)
        out = apply_mistral_rope(q, torch.tensor(cos), torch.tensor(sin))
        dump_vector(f, f"rope{pos}", out)


def dump_rmsnorm(f, cfg):
    torch.manual_seed(42)
    x = torch.randn(cfg["hidden_size"])
    g = torch.randn(cfg["hidden_size"])
    y = rmsnorm(x, g, cfg["norm_eps"])
    dump_vector(f, "norm_x", x)
    dump_vector(f, "norm_g", g)
    dump_vector(f, "norm_y", y)


def dump_mlp(f, model_dir, cfg):
    torch.manual_seed(0)
    h = torch.randn(cfg["hidden_size"])
    gate = load_weight(model_dir, "model.layers.0.mlp.gate_proj.weight")
    up = load_weight(model_dir, "model.layers.0.mlp.up_proj.weight")
    down = load_weight(model_dir, "model.layers.0.mlp.down_proj.weight")
    hidden = silu(F.linear(h, gate)) * F.linear(h, up)
    out = F.linear(hidden, down)
    dump_vector(f, "mlp_h", h)
    dump_vector(f, "mlp_output", out)


def dump_lmhead(f, model_dir, cfg):
    torch.manual_seed(0)
    x = torch.rand(cfg["hidden_size"])
    weight = load_weight(model_dir, "lm_head.weight")
    dump_vector(f, "lmhead_x", x)


def load_layer0_weights(model_dir):
    prefix = "model.layers.0"
    return {
        "q": load_weight(model_dir, f"{prefix}.self_attn.q_proj.weight"),
        "k": load_weight(model_dir, f"{prefix}.self_attn.k_proj.weight"),
        "v": load_weight(model_dir, f"{prefix}.self_attn.v_proj.weight"),
        "o": load_weight(model_dir, f"{prefix}.self_attn.o_proj.weight"),
        "gate": load_weight(model_dir, f"{prefix}.mlp.gate_proj.weight"),
        "up": load_weight(model_dir, f"{prefix}.mlp.up_proj.weight"),
        "down": load_weight(model_dir, f"{prefix}.mlp.down_proj.weight"),
        "input_norm": load_weight(model_dir, f"{prefix}.input_layernorm.weight"),
        "post_attn_norm": load_weight(model_dir, f"{prefix}.post_attention_layernorm.weight"),
    }


def attention_forward(hidden, pos, k_cache, v_cache, weights, cfg):
    inv_freq = compute_inv_freq(cfg["rope_theta"], cfg["head_dim"])
    cos, sin = compute_cos_sin(inv_freq, pos, cfg["head_dim"])
    cos_t = torch.tensor(cos)
    sin_t = torch.tensor(sin)

    q = F.linear(hidden, weights["q"])
    k = F.linear(hidden, weights["k"])
    v = F.linear(hidden, weights["v"])

    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    head_dim = cfg["head_dim"]
    groups = n_heads // n_kv

    q = q.view(n_heads, head_dim)
    k = k.view(n_kv, head_dim)
    v = v.view(n_kv, head_dim)

    q_flat = q.reshape(-1)
    k_flat = k.reshape(-1)
    q_rot = apply_mistral_rope(q_flat, cos_t, sin_t, n_heads, head_dim).view(n_heads, head_dim)
    k_rot = apply_mistral_rope(k_flat, cos_t, sin_t, n_kv, head_dim).view(n_kv, head_dim)

    if k_cache is None:
        k_cache = torch.zeros(n_kv, pos + 1, head_dim)
        v_cache = torch.zeros(n_kv, pos + 1, head_dim)
    elif k_cache.shape[1] <= pos:
        pad = pos + 1 - k_cache.shape[1]
        k_cache = torch.cat([k_cache, torch.zeros(n_kv, pad, head_dim)], dim=1)
        v_cache = torch.cat([v_cache, torch.zeros(n_kv, pad, head_dim)], dim=1)

    k_cache[:, pos, :] = k_rot
    v_cache[:, pos, :] = v

    scale = 1.0 / math.sqrt(head_dim)
    context = torch.zeros(n_heads, head_dim)
    for h in range(n_heads):
        kv_h = h // groups
        q_head = q_rot[h]
        k_heads = k_cache[kv_h, : pos + 1, :]
        v_heads = v_cache[kv_h, : pos + 1, :]
        scores = (k_heads @ q_head) * scale
        scores = torch.softmax(scores, dim=0)
        context[h] = scores @ v_heads

    out = F.linear(context.reshape(-1), weights["o"])
    return q_rot, k_rot, v, out, k_cache, v_cache


def layer_forward(hidden, pos, weights, cfg):
    inv_freq = compute_inv_freq(cfg["rope_theta"], cfg["head_dim"])
    residual = hidden.clone()
    hidden = rmsnorm(hidden, weights["input_norm"], cfg["norm_eps"])
    q, k, v, attn_out, _, _ = attention_forward(
        hidden, pos, None, None, weights, cfg
    )
    hidden = attn_out + residual

    residual = hidden.clone()
    hidden = rmsnorm(hidden, weights["post_attn_norm"], cfg["norm_eps"])
    gate = silu(F.linear(hidden, weights["gate"]))
    up = F.linear(hidden, weights["up"])
    hidden = F.linear(gate * up, weights["down"])
    hidden = hidden + residual
    return hidden, q, k, v, attn_out


def dump_attention(f, model_dir, cfg):
    weights = load_layer0_weights(model_dir)
    torch.manual_seed(0)
    hidden_states = [torch.randn(cfg["hidden_size"]) for _ in range(3)]

    k_cache = v_cache = None
    pos = 0
    for i, hidden in enumerate(hidden_states, start=1):
        dump_vector(f, f"attn_h{i}", hidden)
        q, k, v, out, k_cache, v_cache = attention_forward(
            hidden, pos, k_cache, v_cache, weights, cfg
        )
        dump_vector(f, f"attn_q{i}", q.reshape(-1))
        dump_vector(f, f"attn_k{i}", k.reshape(-1))
        dump_vector(f, f"attn_v{i}", v.reshape(-1))
        dump_vector(f, f"attn_o{i}", out)
        pos += 1


def dump_layer(f, model_dir, cfg):
    weights = load_layer0_weights(model_dir)
    torch.manual_seed(0)
    hidden_states = [torch.randn(cfg["hidden_size"]) for _ in range(3)]
    for i, hidden in enumerate(hidden_states, start=1):
        dump_vector(f, f"layer_h{i}", hidden)
        out, _, _, _, _ = layer_forward(hidden, 0, weights, cfg)
        dump_vector(f, f"layer_o{i}", out)


def dump_module_goldens(model_dir):
    cfg = load_config(model_dir)
    with open(EXPECTED_OUT, "w") as f:
        f.write("# Regenerate: python scripts/test/mistral/goldens.py\n\n")
        dump_rotary_and_rope(f, cfg)
        dump_rmsnorm(f, cfg)
        dump_mlp(f, model_dir, cfg)
        dump_lmhead(f, model_dir, cfg)
        dump_attention(f, model_dir, cfg)
        dump_layer(f, model_dir, cfg)
    print("Wrote", EXPECTED_OUT)


def dump_topk(f, prefix, step, ids, vals):
    dump_vector(f, f"{prefix}_step{step}_top{TOPK}_ids", torch.tensor(ids, dtype=torch.float32))
    dump_vector(f, f"{prefix}_step{step}_top{TOPK}_vals", torch.tensor(vals, dtype=torch.float32))


def topk(logits):
    vals, ids = torch.topk(logits, TOPK)
    return ids.tolist(), [float(v) for v in vals.tolist()]


@torch.inference_mode()
def dump_logits_trace(f, model, input_ids, prefix):
    seq = input_ids
    for step in range(NUM_GEN_STEPS + 1):
        attention_mask = torch.ones_like(seq)
        out = model(seq, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0, -1]
        kid, kval = topk(logits)
        dump_topk(f, f"logits_{prefix}", step, kid, kval)
        next_id = torch.tensor([[kid[0]]], dtype=seq.dtype, device=seq.device)
        seq = torch.cat([seq, next_id], dim=1)


@torch.inference_mode()
def dump_layer_stack(f, model, input_ids, prefix):
    attention_mask = torch.ones_like(input_ids)
    pre_norm = {}

    def capture_pre_norm(_module, inp, _output):
        pre_norm["last"] = inp[0][0, -1].detach().float().cpu()

    handle = model.model.norm.register_forward_hook(capture_pre_norm)
    out = model.model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    handle.remove()

    n_layers = model.config.num_hidden_layers
    for layer in range(n_layers):
        if layer == n_layers - 1:
            dump_vector(f, f"layer_stack_{prefix}_L{layer}", pre_norm["last"])
        else:
            dump_vector(f, f"layer_stack_{prefix}_L{layer}", out.hidden_states[layer + 1][0, -1])
    dump_vector(f, f"layer_stack_{prefix}_norm", out.last_hidden_state[0, -1])


@torch.inference_mode()
def dump_logits_goldens(model_dir):
    device = os.environ.get("LOGITS_DEVICE")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = DTYPES[os.environ.get("LOGITS_DTYPE", "bfloat16")]
    print(f"Loading model from {model_dir} on {device} as {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    with open(LOGITS_OUT, "w") as f:
        f.write(f"# Golden values from Hugging Face Mistral-7B-v0.1 ({dtype})\n")
        f.write("# Regenerate: python scripts/test/mistral/goldens.py\n\n")
        for prefix, prompt in PROMPTS.items():
            print(f"Processing {prefix!r}", flush=True)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            dump_logits_trace(f, model, input_ids, prefix)
            f.flush()
            if DUMP_LAYER_STACK:
                dump_layer_stack(f, model, input_ids, prefix)
                f.flush()
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    del model
    gc.collect()
    print("Wrote", LOGITS_OUT)


def main():
    if not os.path.isdir(MODEL):
        raise SystemExit(f"Model directory not found: {MODEL}")

    dump_module_goldens(MODEL)
    if os.environ.get("GOLDENS_MODULE_ONLY", "0") != "1":
        dump_logits_goldens(MODEL)


if __name__ == "__main__":
    main()
