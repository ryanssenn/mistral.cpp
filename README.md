# qmog.cpp

**Experimental project under active development**.

A compact C++ inference engine optimized for Apple platforms.

Load a single `.mog` (Model Object Graph) file and run inference locally. No runtime dependencies. A small C++ codebase focused on readability and simplicity.

## Supported models

| Model | Format | Size |
| ----- | ------ | ---- |
| [Qwen3-0.6B f16](https://huggingface.co/QmogAI/Qwen3-0.6B.mog) | MOG v2, f16 | ~1.2 GB |

## Run it

Only available on MacOS.

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

3. Pull the `.mog` model:

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

Perplexity is the main correctness check. `./perplexity.sh` runs the engine against a Hugging Face reference on a fixed prompt. Use `--save` to record the current numbers as the baseline in `perplexity_baseline.json`.

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
