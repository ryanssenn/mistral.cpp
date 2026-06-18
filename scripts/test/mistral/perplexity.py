"""
HF fp32 reference perplexity for a single prompt.

Computes teacher-forced perplexity = exp(mean -log softmax(logits)[next_token])
over the prompt tokens, using the Hugging Face Mistral implementation. This is
the reference target the int8 engine's own perplexity is driven down toward;
the two numbers are produced independently (each model scores the prompt's real
next tokens), not by diffing logits against each other.

Usage (from repo root, with ../Mistral-7B-v0.1 present):
  python scripts/test/mistral/perplexity.py "<prompt>"

The prompt may also be piped on stdin if no argument is given. Env overrides:
  MISTRAL_MODEL   path to the HF checkout (default: ../Mistral-7B-v0.1)
  PPL_DTYPE       float32 (default) | bfloat16 | float16
  PPL_DEVICE      cpu (default) | mps | cuda
"""
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL = os.environ.get(
    "MISTRAL_MODEL",
    os.path.abspath(os.path.join(REPO_ROOT, "../Mistral-7B-v0.1")),
)


def read_prompt():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit('Usage: python scripts/test/mistral/perplexity.py "<prompt>"')


@torch.inference_mode()
def main():
    prompt = read_prompt().rstrip("\n")
    if not prompt:
        raise SystemExit("Empty prompt")
    if not os.path.isdir(MODEL):
        raise SystemExit(f"Model directory not found: {MODEL}")

    device = os.environ.get("PPL_DEVICE", "cpu")
    dtype = DTYPES[os.environ.get("PPL_DTYPE", "float32")]
    print(f"loading HF model from {MODEL} on {device} as {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()

    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ids = torch.tensor([token_ids], device=device)
    logits = model(ids).logits[0].float()  # [N, vocab]

    logprobs = torch.log_softmax(logits, dim=-1)
    nlls = [float(-logprobs[i, token_ids[i + 1]]) for i in range(len(token_ids) - 1)]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")

    # Machine-parsable lines (consumed by perplexity.sh).
    print("token_ids:", " ".join(str(t) for t in token_ids))
    print("tokens:", len(token_ids))
    print(f"perplexity: {ppl}")


if __name__ == "__main__":
    main()
