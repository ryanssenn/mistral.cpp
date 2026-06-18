# mistral.cpp

A from-scratch C++ implementation of Mistral 7B inference on CPU. The core was hand-written through the first successful forward pass; agents were then used to accelerate the rest. Validated against Hugging Face reference outputs, with f32 and int8 paths.

Measured on an AWS Linux instance (16 vCPU Intel Xeon Platinum 8488C, 32 GiB RAM), the int8 implementation achieves ~2.6 tok/s throughput with a perplexity of ~5.2, matching the Hugging Face reference.

An educational project: compact code you can read through, not a production engine. Not affiliated with Mistral AI.

# Running

Only validated on macOS and Linux.

| | Minimum |
|---|---|
| RAM | 16 GiB |
| Disk | 40 GiB |
| Python | 3.10+ |
| CMake | 3.20+ |
| Compiler | C++17 (gcc 11+ or clang) |

These commands assume the Mistral Hugging Face checkout and this repo are sibling directories:

```text
parent-directory/
  Mistral-7B-v0.1/
  mistral.cpp/
```

#### 1. Download Mistral 7B v0.1 and mistral.cpp

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/mistral.cpp.git
cd mistral.cpp
```

If the model download fails, make sure your Hugging Face account has access to `mistralai/Mistral-7B-v0.1` and that Git LFS is installed.

#### 2. Create the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Export the model binary expected by the C++ app and tests

The app and tests read quantization mode from the binary header, so export the format you want. Both paths write to `./mistral.bin` in the repo root (re-export to switch formats).

**Default - int8 (smaller, faster inference)**

Per-group symmetric int8 quantization on MLP gate/up weights (~18 GB). `down_proj`, attention, embeddings, norms, and `lm_head` stay f32 for generation quality.

```bash
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin
```

**Option - f32 (full parity tests)**

Full-precision weights (~27 GB). Best for validating correctness against Hugging Face. Runs 21 parity tests.

```bash
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant f32
```

Expected result (either option):

```text
Completed
```

#### 4. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Expected result:

```text
Built target mistral.cpp
Built target test_exec
```

#### 5. Run text completion

Same command for f32 and int8 - the runtime picks the path from the binary header.

```bash
./build/mistral.cpp ./mistral.bin "Paris is the capital of" --temp 0.7
```

`--temp` controls sampling: `0` is greedy (default), values like `0.7` add randomness. A repetition penalty is applied on every step regardless of temperature, so even greedy decoding discourages repeats.

The program prints up to 50 generated tokens and then a throughput line like:

```text
throughput: <number> tok/s
```

The default int8 export is much faster on CPU than f32. Use f32 mainly for correctness checks.

# Testing

The primary quality metric is perplexity on a fixed prompt (lower is better); the int8 engine's perplexity should fall toward the Hugging Face reference as numerical bugs are fixed. `perplexity.sh` runs both on the same prompt and compares against a saved baseline so regressions are visible:

```bash
./perplexity.sh                 # int8 engine PPL + cached HF reference + delta vs baseline
./perplexity.sh --hf            # also recompute the HF reference (slow; loads the 7B model)
./perplexity.sh --save          # record current numbers as the new baseline after an improvement
./perplexity.sh "your own prompt"
```

Example:

```text
====================================================
  perplexity            prompt sha: 1854ad2a801c
====================================================
  HF reference        5.2414
  int8 engine         32.9683
  gap (int8 - HF)     27.7269
  vs baseline int8    32.9683  (+0.0000, unchanged)
====================================================
```

The HF reference defaults to fp32; on memory-constrained machines set `PPL_DTYPE=bfloat16` (Mistral's native dtype) to avoid loading a ~28 GB f32 copy. The engine on its own (no Hugging Face) is just `./build/mistral.cpp ./mistral.bin "<prompt>" --ppl`.

## Component tests

A unit test suite validates tokenizer behavior, CPU kernels, decoder modules, hidden states, and logits against Hugging Face reference tensors for both f32 and int8 exports. After creating `./mistral.bin`, build and run the tests from the repository root:

```bash
cmake --build build --target test_exec
./build/test_exec
```

The f32 export includes 21 parity tests, while the int8 export includes 7. One test, `test logits multi top10`, is expected to fail for int8 because it highlights top-1 token differences relative to the f32 reference. Perplexity is a more meaningful measure of overall int8 model quality.

If you encounter `Model binary open failed`, `./mistral.bin` is missing from the repository root. Generate it using the export step above.


# Resources

Reading and reference material used while building mistral.cpp.

### Machine learning theory

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original transformer paper
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) - Andrej Karpathy
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU) - RoPE walkthrough

### Systems and performance

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) - Edward Z. Yang
- [C++ Vtables](https://shaharmike.com/cpp/vtable-part1/) - Shahar Mike
- [yalm](https://andrewkchan.dev/posts/yalm.html) - Andrew Chan
- [LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/) - Arseny Kapoulkine
- [Quantize llama models with ggml and llama.cpp](https://medium.com/data-science/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172) - Maxime Labonne

### Reference implementations

- [Hugging Face Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [calm](https://github.com/zeux/calm/tree/main) - Arseny Kapoulkine
- [llama.cpp](https://github.com/ggml-org/llama.cpp/) - Georgi Gerganov
- [llama2.c export and quantization](https://github.com/karpathy/llama2.c/blob/master/export.py) - Andrej Karpathy

