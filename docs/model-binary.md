# MOG format

MOG (Model Object Graph) is the single-file format loaded by qmog.cpp. One `.mog` file contains the runtime config, tokenizer data, tensor index, and raw weight payload.

The C++ loader is `Parameters::load_parameters()` in `src/loader/parameters.cpp`. It memory-maps the file and exposes tensor views into the payload. The weight bytes are not copied into heap memory.

## File layout

A `.mog` file has four sections:

```text
file prefix     fixed 16 bytes: magic, version, header size
header          architecture, config, tokenizer, tensor table
padding         0-63 bytes so the payload starts on a 64-byte boundary
payload         raw tensor bytes
```

The header is metadata. The payload is the large part of the file. Tensor entries in the header store byte offsets into the payload, so loading a tensor is just pointer arithmetic plus a `Tensor<T>` wrapper.

## File prefix

All multi-byte integers are little-endian.

| Offset | Size | Field | Meaning |
| ------ | ---- | ----- | ------- |
| 0 | 4 | magic | `MOG\0` |
| 4 | 4 | version | `1` or `2` |
| 8 | 8 | header_size | header length in bytes |

The current loader accepts versions `1` through `2`.

Payload start:

```text
payload_base = align_up(16 + header_size, 64)
```

If the magic, version, header size, or payload start is invalid, the loader exits.

## Header encoding

Strings are encoded as:

```text
u32 byte_length
UTF-8 bytes
```

The header is read in this order:

1. architecture string
2. config key-value table
3. tokenizer data
4. tensor table

## Architecture

The first header value is a string such as:

```text
qwen3
```

The loader stores this in `Config::architecture`. The Qwen tokenizer implementation is fixed to Qwen3 behavior in `QwenTokenizer`.

## Config

Config is a key-value table:

```text
u32 count
repeat count times:
    string key
    u8 type
    value
```

Supported value types:

| Type byte | Meaning | Encoding |
| --------- | ------- | -------- |
| 0 | string | length-prefixed string |
| 1 | uint32 | `u32` |
| 2 | float32 | `f32` |

Current runtime keys:

| Key | Type | Example for Qwen3-0.6B |
| --- | ---- | ---------------------- |
| `hidden_size` | uint32 | 1024 |
| `intermediate_size` | uint32 | 3072 |
| `n_layers` | uint32 | 28 |
| `n_heads` | uint32 | 16 |
| `n_kv_heads` | uint32 | 8 |
| `head_dim` | uint32 | 128 |
| `vocab_size` | uint32 | 151936 |
| `sliding_window` | uint32 | 0 |
| `max_position_embeddings` | uint32 | 40960 |
| `rope_theta` | float32 | 1000000.0 |
| `norm_eps` | float32 | 1e-6 |
| `quant` | string | `f16` |
| `tie_word_embeddings` | uint32 | 1 |
| `bos_token_id` | uint32 | 151643 |
| `eos_token_id` | uint32 | 151645 |

Unknown keys are skipped by type. This lets newer exporters add metadata without breaking older loaders, as long as the value type is known.

If `head_dim` is not present, the loader computes it as:

```text
hidden_size / n_heads
```

## Tokenizer data

Tokenizer data starts with the vocabulary:

```text
u32 vocab_count
repeat vocab_count times:
    string token
    u32 id
```

The loader builds:

- `token_to_id`
- `id_to_token`

Then it reads BPE merge rules:

```text
u32 merge_count
repeat merge_count times:
    string merge_rule
```

Each merge rule contains two token strings separated by a space. The loader converts those token strings into ids, packs the pair into a `uint64_t`, and stores:

- `merge_to_rank[pair]`
- `merge_to_id[pair]`

For MOG v2, tokenizer data also includes a pre-tokenization regex string after the merge table:

```text
string pre_tokenize_regex
```

Qwen tokenization uses this regex for tiktoken-style pre-tokenization.

## Tensor table

The tensor table is an index over the payload:

```text
u32 count
repeat count times:
    string name
    u8 dtype
    u8 ndim
    u32 dims[4]
    u64 offset
    u64 scale_offset
    u32 scale_size
```

Tensor dtypes:

| Type byte | Runtime tensor |
| --------- | -------------- |
| 0 | `Tensor<float>` |
| 1 | `Tensor<int8_t>` |
| 2 | `Tensor<fp16_t>` |

For f32 and f16 tensors, `scale_offset` and `scale_size` are zero.

For int8 tensors, `scale_offset` points to f32 scale values in the payload and `scale_size` is the number of scales.

## Tensor routing

Tensor names are stored using Hugging Face-style names.

Layer tensors use this prefix:

```text
model.layers.{layer}.
```

The loader strips that prefix and stores the tensor in:

```text
layer_weights[layer][suffix]
```

Example:

```text
model.layers.0.self_attn.q_proj.weight
```

becomes:

```text
layer_weights[0]["self_attn.q_proj.weight"]
```

Non-layer tensors are stored in:

```text
global_weights[name]
```

Examples:

- `model.embed_tokens.weight`
- `model.norm.weight`

For models with tied output embeddings, there may be no separate `lm_head.weight`.

## Payload

The payload is raw tensor data. Tensor offsets are relative to `payload_base`, not the beginning of the file.

Each tensor is contiguous in row-major order. Exporters may add padding between tensors so the next tensor begins at an aligned offset.

The loader wraps payload memory directly:

```cpp
Tensor<float>(float* at offset, shape)
Tensor<fp16_t>(fp16_t* at offset, shape)
Tensor<int8_t>(int8_t* at offset, scales, shape)
```

The mmap must stay alive for the lifetime of these tensor views.

## Load flow

```text
open file
mmap file
check prefix
read architecture
read config
read tokenizer data
compute payload_base
read tensor table
create tensor views into payload
```

After load, `Parameters` owns:

- `config`
- `tokenizer`
- `global_weights`
- `layer_weights`

The runtime uses `get_tensor(layer, name)` to retrieve typed tensor views from those maps.

## Quick reference

| Topic | Detail |
| ----- | ------ |
| Magic | `MOG\0` |
| Supported versions | `1`, `2` |
| Header encoding | little-endian binary |
| String encoding | `u32` byte length + UTF-8 |
| Payload alignment | 64 bytes |
| Tensor dtypes | f32, int8, f16 |
| Current Qwen file | MOG v2, f16 |
