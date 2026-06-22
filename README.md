# qmog.cpp

**Experimental project under active development**. Current support is limited to the Mistral 7B base model running on a CPU backend. Metal acceleration, additional architectures, and performance improvements are ongoing.

A compact C++ inference engine optimized for Apple platforms.

Load a single `.mog` (Model Object Graph) file and run inference locally. No runtime dependencies. A small C++ codebase focused on readability and simplicity.

<img width="800" height="245" alt="qmog_demo" src="https://github.com/user-attachments/assets/0d7804dc-bbb6-4970-8a1f-35ab30add8cf" />

## Supported models

Benchmarks on M4 MacBook, Q8F16 (~10 GB).


| Model                                                                   | tok/s | perplexity |
| ----------------------------------------------------------------------- | ----- | ---------- |
| [Mistral-7B-v0.1 Q8F16](https://huggingface.co/QmogAI/Mistral-7B-Q8F16) | 5.73  | 5.24       |


## Run it

Built and tested on macOS. Linux is not supported.

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
hf download QmogAI/Mistral-7B-Q8F16 mistral-7B-Q8F16.mog --local-dir .
```

4. Run:

```bash
./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0.7
```

Use `--temp 0` for greedy decoding.

To export your own `.mog` from a Hugging Face checkpoint, use [qpack](https://github.com/ryanssenn/qpack).

## Testing

Perplexity is the main correctness check. `./perplexity.sh` runs the engine against a Hugging Face reference on a fixed prompt. Regenerate unit-test goldens with `python scripts/test/mistral/goldens.py`.

```bash
./perplexity.sh
./perplexity.sh --check
./build/test_exec
```

