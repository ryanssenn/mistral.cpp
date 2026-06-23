# Optimizations

This document tracks enabled CPU optimizations and their measured impact on Mistral-7B Q8F16.

## F16 matmul accumulation

Implementation:

- Applies to `matmul<fp16_t, float, float>` in the CPU backend.
- Converts f32 activation chunks to f16 inside the row kernel.
- Uses NEON f16 FMA (`vfmaq_f16`) to accumulate f16-weight dot products in f16 lanes.
- Converts the final row result back to float, so downstream tensors and module APIs stay unchanged.

Benefit:

- Uses wider native half-precision vector arithmetic on Apple Silicon.
- Avoids widening every f16 weight to f32 before multiplication.
- Improves throughput with a small accepted perplexity movement.

Results:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.24101`, `tokens: 33`, `delta=+0.0029` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `6.62252 tok/s` vs baseline `5.11822 tok/s` |

## Int8 ARM dot-product matmul

Implementation:

- Applies to `matmul<int8_t, float, float>` in the CPU backend.
- Quantizes the f32 activation vector to int8 once per int8 matmul.
- Uses per-64-element activation scales aligned with the existing per-group int8 weight scales.
- Uses ARM dot-product instructions (`vdotq_s32`) to compute int8 x int8 dot products into int32 lanes.
- Applies weight and activation dequantization scales before writing float row outputs.

Benefit:

- Targets MLP `gate_proj` and `up_proj`, the largest repeated matmuls in Q8F16 decode.
- Replaces int8-to-f32 widening plus f32 FMA with native int8 dot-product instructions.
- Keeps the test suite passing while improving measured throughput.

Results:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.24294`, `tokens: 33`, `delta=+0.0049` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `7.88188 tok/s` vs Release control `7.12923 tok/s` |

