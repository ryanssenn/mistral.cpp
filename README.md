# mog.cpp

A compact C++ inference engine optimized for Apple platforms.

Convert a Hugging Face model into a single .mog file and run it locally. No runtime dependencies. A small C++ codebase focused on readability and simplicity.

## Supported models

Benchmarks on M4 MacBook, Q8F16 (~10 GB).

| Model | tok/s | perplexity |
|-------|------:|-----------:|
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 5.73 | 5.24 |

## Run it

Built and tested on macOS. Linux builds are supported but not the primary target.

1. Clone the model and this repo into the same parent directory:

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/mog.cpp.git
cd mog.cpp
```

2. Build a `.mog` file. The engine reads a single `.mog` (Model Object Graph) file with config, tokenizer, and weights. Use `export_mistral.py` to convert the Hugging Face checkpoint:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral.mog
```

3. Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

4. Run:

```bash
./build/mog-cli ./mistral.mog "Paris is the capital of" --temp 0.7
```

Use `--temp 0` for greedy decoding.

## Testing

Perplexity is the main correctness check. `./perplexity.sh` runs the engine against a Hugging Face reference on a fixed prompt. Regenerate unit-test goldens with `python scripts/test/mistral/goldens.py`.

```bash
./perplexity.sh
./build/test_exec
```
