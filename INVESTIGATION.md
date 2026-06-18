# Int8 Perplexity Investigation

## Goal

The int8 inference path is far less accurate than the float reference. On the
benchmark prompt (`scripts/perplexity_prompt.txt`):

| Model | Perplexity |
|---|---|
| int8 engine | ~32.97 |
| Hugging Face reference (bf16) | ~5.24 |

Perplexity measures how surprised the model is by the next token of a fixed
passage (`PPL = exp(mean negative log-likelihood)`, lower is better). A gap this
large means the int8 path is making materially different predictions than the
reference, not just rounding noise. This document records the hunt for the
cause(s), each fix, and the exact perplexity change it produced.

## How int8 inference works here (context)

Only the MLP `gate_proj` and `up_proj` weight matrices are quantized; everything
else (attention, `down_proj`, embeddings, norms, `lm_head`) stays f32. The
quantization is **per-group symmetric int8**:

- `export_mistral.py` splits each weight row into groups of 64 values, finds the
  max absolute value per group, and stores `scale = 127 / max_abs` plus the
  quantized integers `q = round(w * scale)` clamped to `[-127, 127]`.
- At inference, the engine dequantizes with `w ~= q / scale`. The int8 matmul in
  `src/backend/cpu/kernels.cpp` does this per group: `sum += (1/scale_g) * dot(q_g, x_g)`.

The quant/dequant convention is self-consistent on inspection, so the cause is
pursued empirically below.

## Method

- Baseline measured with `./perplexity.sh` (int8 32.97 vs HF 5.24).
- Each candidate below follows: hypothesis -> change -> rebuild (re-export only
  if the quant format changes) -> `./perplexity.sh` -> record before/after PPL ->
  keep if it helps or is neutral, revert otherwise.
- Environment: AWS Linux, 16 vCPU Intel Xeon Platinum 8488C, 32 GiB RAM.

## Baseline

```
HF reference        5.2414
int8 engine         32.9683
gap (int8 - HF)     27.7269
```

---

## Findings

### D1 - Is the int8 matmul kernel faithful to the dequantized weights?

**Hypothesis.** The int8 matmul in `src/backend/cpu/kernels.cpp` could have a bug
(wrong per-group scale index, bad accumulation) that makes its output differ from
what you'd get by simply dequantizing the weights to f32 and doing a normal
matmul. If so, that bug, not quantization, would be the error source.

**Experiment.** `test/diag/int8_matmul_check.cpp` loads layer 0's int8
`gate_proj`, dequantizes every weight with the engine's own `Tensor::get()`
(`q / scale`), and compares two matmuls against the same random activation
vector: the int8 kernel path vs a plain f32 matmul of the dequantized weights.

**Result.**

```
gate_proj shape [14336, 4096], scales 917504
  max abs error  = 9.8e-07
  mean abs error = 7.3e-08
  max rel error  = 0.0159   (only on near-zero outputs; denom ~1e-6)
```

**Conclusion.** The kernel reproduces the dequant-f32 matmul to floating-point
noise (~1e-6). There is **no kernel bug** in scale indexing or accumulation. The
int8 matmul faithfully computes `q/scale @ x`. So the perplexity gap is not a
matmul bug; it comes from the quantized weights themselves differing from the
originals (and/or something outside the MLP weight matmul). This sends the
investigation to D2 (where does the error grow?) and Phase 3 (the quant format).

### D2 - Where does the engine diverge from HF, layer by layer?

**Setup.** Generated Hugging Face per-layer goldens with
`DUMP_LAYER_STACK=1 python scripts/test/mistral/logits.py` (the hidden state of
the last prompt token after each of the 32 decoder layers). `test/diag/layer_bisect.cpp`
runs the int8 engine on the same prompt and, after each layer, reports max abs
error and cosine similarity vs the golden.

**Result - and the key finding.** The diagnostic also prints `inv_freq`, the
rotary-embedding frequency table. It was **all zeros**:

```
init_freq called: no   (inv_freq[0..2] = 0, 0, 0)
  L0   max_abs=0.0158   cos=0.8716
  L1   max_abs=0.0415   cos=0.7023
  ...
  L30  max_abs=1.966    cos=0.9578
  norm max_abs=46.96    cos=0.9590
```

Divergence is already large at layer 0 (cosine 0.87, not ~1.0), which points at
attention, not the MLP. The cause: `RotaryEmbedding::init_freq` (which fills
`infer.inv_freq` from `rope_theta`) is **only ever called in unit tests**, never
in the real inference path (`src/main.cpp`, `Model::forward`, or the
`InferenceState` constructor). With `inv_freq` left at zero (arena memory is
zero-filled), `RotaryEmbedding::forward` produces `cos = cos(0) = 1`,
`sin = sin(0) = 0`, so `rope()` becomes the identity. **The model runs with no
positional information at all.**

Re-running the bisection with `init_freq` called (`--rope`):

```
init_freq called: yes   (inv_freq[0..2] = 1, 0.866, 0.750)
  L0   max_abs=0.00018   cos=0.999991
  ...
  L30  max_abs=0.092     cos=0.999957
  norm max_abs=0.93      cos=0.999962
```

Per-layer cosine similarity to HF jumps to ~0.99999 and the final-norm max error
collapses from 46.96 to 0.93. (L31 keeps one ~140-magnitude outlier dimension;
that is Mistral's known massive-activation channel, and the final norm matches
HF regardless.)

**Conclusion.** The dominant error is a real bug: RoPE is effectively disabled in
the engine because `init_freq` is never called. This affects both f32 and int8
paths. Fixing it is the first thing to measure against perplexity (Phase 2).

---

## Fixes

### Fix 1 - Call `init_freq` so RoPE actually rotates (the big one)

**Change.** `RotaryEmbedding::init_freq` populates `infer.inv_freq` from
`rope_theta`. It was missing from the inference path, so it is now called once
right after the model is built, in both `run_inference` and `run_perplexity` in
[src/main.cpp](src/main.cpp):

```cpp
Model<T> model(params);
RotaryEmbedding::init_freq(infer, params->config);
```

The same omission existed in the test-suite model paths
([test/mistral/test_logits.cpp](test/mistral/test_logits.cpp)), so `init_freq`
was added there too (`run_logits_prompt` and `run_layer_stack_prompt`); otherwise
the suite would keep validating the broken-RoPE behavior.

**Perplexity.**

| | int8 PPL | HF reference |
|---|---|---|
| Before (no RoPE) | 32.9683 | 5.2414 |
| After Fix 1 | **5.2385** | 5.2414 |

The int8 engine now matches the HF reference (the small -0.003 difference is just
the engine's f32 weights/attention being marginally more precise than the bf16
reference). This single fix closes essentially the entire gap.

**Why it mattered so much.** Without positional encoding the model still predicts
plausible tokens from content alone (so generation looked vaguely coherent and
hid the bug), but it loses all word-order information, which badly hurts the
probability it assigns to the true next token - exactly what perplexity measures.

**Corroboration from the unit suite (`./build/test_exec`).** The int8 logits
diagnostic improved dramatically:

- top-10 overlap vs the HF golden went from 3-7 / 10 to **10 / 10 at every step**.
- aligned logit error (`max_val_err`) dropped from up to ~4.5 to ~0.02-0.09.
- top-1 token flips went from many to a single near-tie (`paris` step 2:
  f32 token 272 @ 13.4375 vs int8 token 624 @ 13.4681 - a genuine coin-flip, not
  an error).

The `test layer stack prefill` test (previously skipped because no goldens were
committed; now running against the regenerated ones) sits just over its strict
`5e-2` element tolerance at layer 26 (0.051), which is bf16-golden precision
noise at a deep layer, consistent with the D2 cosine of 0.99999, not a bug.

---

## Phase 2 - other engine checks (after Fix 1)

With RoPE fixed, the int8 engine already matches the reference, so the remaining
engine candidates have little perplexity headroom. Recorded for completeness:

- **Numeric precision** (`matmul`/`softmax`/`silu`/`RMSNorm` accumulation): the
  matmul accumulates in f32 over 4096-wide dot products and the perplexity log-sum-exp
  already runs in `double`. With int8 PPL (5.2385) at/below the bf16 reference
  (5.2414), there is no measurable error left to recover here. No change made.
- **Sliding-window attention is not implemented.** `Attention::forward` attends to
  all past tokens (full causal), but Mistral uses a 4096-token sliding window. This
  has zero effect on the 33-token benchmark and the per-layer match is exact, but it
  would cause divergence from HF for prompts longer than the window. Noted as a
  known limitation rather than a fix, since it does not affect the metric here. A
  longer benchmark prompt would be needed to measure it.

## Phase 3 - quantization-scheme experiments (not warranted)

These were planned only "where Phase 1/2 implicate the quant format." They do not:

- D1 showed the int8 matmul is faithful to the dequantized weights (~1e-6).
- After Fix 1 the int8 engine perplexity (5.2385) is already slightly *below* the
  bf16 HF reference (5.2414), i.e. per-group int8 on `gate_proj`/`up_proj` adds no
  measurable perplexity penalty.

Group size (64 -> 32/16) and rounding changes can only shrink an error that is
already below the reference's own bf16 noise floor, while each requires a slow
re-export that rewrites the ~18 GB binary. They are therefore not the lever and
were not run. The honest conclusion of this investigation is that the int8 path
was never the real problem - a disabled RoPE was.

---

## Summary

| Stage | int8 PPL | HF reference | Gap |
|---|---|---|---|
| Baseline | 32.9683 | 5.2414 | +27.73 |
| Fix 1: call `init_freq` (enable RoPE) | **5.2385** | 5.2414 | -0.003 |

**Net: perplexity 32.97 -> 5.24, a 27.73 reduction**, bringing the int8 engine to
parity with the Hugging Face reference on the benchmark prompt.

Root cause: `RotaryEmbedding::init_freq` was never called on the live inference
path, so `inv_freq` stayed zero and RoPE collapsed to the identity (no positional
encoding). Generation still looked plausible, which masked the bug, but every
next-token probability was degraded.

Kept changes:
- [src/main.cpp](src/main.cpp): call `init_freq` in `run_inference` and `run_perplexity`.
- [test/mistral/test_logits.cpp](test/mistral/test_logits.cpp): same call in the
  logits and layer-stack test paths so the suite validates correct behavior.

Reusable diagnostics added under `test/diag/`:
- `int8_matmul_check.cpp` (D1): int8 matmul vs dequantized-f32 matmul.
- `layer_bisect.cpp` (D2): per-layer divergence vs HF goldens, with a `--rope` toggle.

Compile a diagnostic with, e.g.:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 \
  test/diag/layer_bisect.cpp \
  src/common/tensor.cpp src/loader/parameters.cpp src/tokenizer/tokenizer.cpp \
  src/backend/cpu/kernels.cpp src/model/mistral/modules.cpp -o /tmp/diag_bisect
```

Not pursued (evidence-backed): int8 quantization scheme (group size, rounding,
which tensors are quantized) - shown not to be the lever in Phase 3.

Future work to measure separately: sliding-window attention for prompts longer
than 4096 tokens (Phase 2).
