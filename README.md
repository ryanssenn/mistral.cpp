# mistral.cpp

From-scratch C++ implementation of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU. It runs local text completion with its own tensors, quantized weight loader, BPE tokenizer, manual memory management with KV caching, and the full decoder architecture.

Educational project, meant for understanding how inference works by reading the code, not as a production inference engine.

Independent project; not affiliated with Mistral AI.

<br>

![demo2](https://github.com/user-attachments/assets/1711dc3e-9ab2-4f73-8c35-b7ac3aabec55)

# Running

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

Per-group symmetric int8 quantization on MLP projection weights (~16 GB). Attention projections, embeddings, norms, and `lm_head` stay f32. Faster on CPU than full f32, with much better output quality than quantizing attention.

```bash
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin
```

**Option - f32 (full parity tests)**

Full-precision weights (~27 GB). Best for validating correctness against Hugging Face. Runs 19 parity tests.

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
./build/mistral.cpp ./mistral.bin "Paris is the capital of"
```

The program prints up to 50 generated tokens and then a throughput line like:

```text
throughput: <number> tok/s
```

The default int8 export is much faster on CPU than f32. Use f32 mainly for correctness checks.

# Testing

To check that the C++ code matches the real Mistral implementation, I validated each component separately rather than only checking end-to-end output.

First, the Python scripts in `scripts/test/mistral/` run individual pieces, attention, RMSNorm, RoPE, MLP, etc. using Hugging Face's Mistral with the actual weights. Each script dumps its output tensors into `test/mistral/expected.txt` as named float arrays.

Then the C++ tests in `test/mistral/` load those values and compare them against the output of the corresponding mistral.cpp code. For example, an attention test copies a known `hidden_state` from the golden file, runs `Attention::forward`, and checks that Q/K/V and the output match. The same pattern is used for the tokenizer, CPU kernels (matmul, softmax, RoPE, SiLU), and each decoder module. Comparisons use a tolerance of ±0.05.

Run the tests from the repo root after creating `./mistral.bin`:

```bash
cmake --build build --target test_exec
./build/test_exec
```

Tests are filtered by the quantization mode in `mistral.bin`. With the default int8 export, tests cover load, MLP, and tokenizer only. Golden values in `test/mistral/expected.txt` come from Hugging Face f32 weights; re-export with `--quant f32` to run the full 19-test suite.

**Expected result (default int8 export):**

```text
Running tests for model type: int8
Running load config
Running load weights
Running test attention feedforward mlp
Running tokenizer encode
Running tokenizer encode fallback

Summary : 5 / 5 tests passed
```

**Expected result (f32 export):**

```text
Running tests for model type: f32
Running test rope
Running test matmul
Running test row matmul
Running test softmax
Running test silu
Running load config
Running load weights
Running test layer
Running test attention
Running test attention feedforward mlp
Running test kv cache
Running test embedding
Running test rotary embedding inv freq
Running test rotary embedding
Running test rmsnorm
Running test lm head
Running tokenizer encode
Running tokenizer encode fallback
Running tokenizer decode

Summary : 19 / 19 tests passed
```

If you see this:

```text
Model binary open failed
```

then `./mistral.bin` does not exist at the repo root. Run the export command in step 3, or copy the exported model binary to `./mistral.bin`.

# Roadmap

Full progress tracker: [ROADMAP.md](ROADMAP.md). Still todo: temperature scaling, terminal chat interface, fp8, SIMD, CUDA.


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

# License

MIT
