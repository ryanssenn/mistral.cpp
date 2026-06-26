# qmog.cpp

**Experimental project under active development**.

A compact C++ inference engine optimized for Apple platforms.

Load a single model binary and run inference locally. No runtime dependencies. A small C++ codebase focused on readability and simplicity.

`.mog` is qmog's model format containing everything needed for inference. Pre-built `.mog` files are available, or you can generate your own from a Hugging Face Qwen3 checkpoint using [qpack](https://github.com/ryanssenn/qpack).

## Supported models

| Model | Size |
| ----- | ---- |
| [Qwen3-0.6B f16](https://huggingface.co/QmogAI/Qwen3-0.6B.mog) | ~1.2 GB |

## Run it

Only available on macOS.

1. Clone this repo:

```bash
git clone https://github.com/ryanssenn/qmog.cpp.git
cd qmog.cpp
```

2. Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

3. Download the model.

Requires the Hugging Face CLI (`pip install "huggingface_hub[cli]"`).

```bash
hf download QmogAI/Qwen3-0.6B.mog qwen3-0.6B.mog --local-dir .
```

4. Run:

```bash
./build/qmog-cli qwen3-0.6B.mog "Hello"
```

Use `--temp 0` for greedy decoding.

To export your own `.mog` from a Hugging Face checkpoint, use [qpack](https://github.com/ryanssenn/qpack).

## Testing

Requires `qwen3-0.6B.mog` in the repo root.

### Perplexity

Perplexity is the main correctness check. It compares qmog against a Hugging Face reference implementation on a fixed prompt to verify the engine produces the same probabilities. Use `--save` to record the current results as the baseline in `perplexity_baseline.json`.

```bash
./perplexity.sh
./perplexity.sh --check
./perplexity.sh --save
```

### Unit tests

Runs all tests under `test/qwen/`.

```bash
./build/test_exec
```