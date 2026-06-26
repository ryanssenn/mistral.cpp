# End-to-end forward pass

This is the story of what happens when you run:

```bash
./build/qmog-cli qwen3-0.6B.mog "Hello"
```

The goal is simple: turn prompt text into token ids, run those ids through the model, produce logits, and pick the next token.

## 1. The model file is opened

The CLI starts in `src/main.cpp`. It reads the model path and prompt from argv, then creates `Parameters` and calls:

```cpp
params->load_parameters(model_path);
```

`Parameters::load_parameters()` lives in `src/loader/parameters.cpp`. It opens the `.mog` file, memory-maps it, checks the magic/version, then parses the header.

The `.mog` file contains:

- model config
- tokenizer vocabulary and merge metadata
- tensor index
- packed model weights

The loader does not copy the full weight payload into heap memory. It creates tensor views that point directly into the mmap'd file.

## 2. Config becomes runtime shape information

The header config fills `Config` in `include/loader/parameters.h`.

This tells the runtime how large every tensor should be:

- `hidden_size`
- `intermediate_size`
- `n_layers`
- `n_heads`
- `n_kv_heads`
- `head_dim`
- `vocab_size`
- `rope_theta`
- `norm_eps`
- `quant`

The same config is later used to allocate `InferenceState`, size logits, shape Q/K/V tensors, and build the KV cache.

## 3. Weights become tensor views

After config and tokenizer data, the loader reads the tensor table.

Each tensor entry says:

- name
- dtype
- shape
- payload offset
- optional quantization scale metadata

Layer weights are routed into `layer_weights[layer]`. Global weights, such as embeddings or final norm, go into `global_weights`.

The tensors are stored as:

```cpp
std::variant<Tensor<float>, Tensor<int8_t>, Tensor<fp16_t>>
```

That lets the same loader represent f32, int8, and f16 weights.

## 4. The prompt becomes token ids

Back in `src/main.cpp`, the prompt string is encoded:

```cpp
std::vector<uint32_t> got = params->tokenizer.encode(text);
```

`QwenTokenizer` is the public wrapper in `include/tokenizer/qwen_tokenizer.h`. Shared BPE logic lives in `include/tokenizer/bpe.h`; regex pre-tokenization lives in `include/tokenizer/qwen_pre_tokenize.h`.

For BPE-style tokenizers, the rough flow is:

1. Apply model-specific pre-tokenization.
2. Convert initial text pieces to token ids.
3. Fall back to byte tokens when needed.
4. Repeatedly merge adjacent pairs by learned rank.
5. Add special tokens if the model expects them.

The result is a list of integer ids. From this point on, the model is not working with strings.

## 5. InferenceState allocates scratch space

The CLI constructs:

```cpp
InferenceState infer(params->config);
```

`InferenceState` is in `include/model/inference_state.h`. It owns the temporary tensors needed for one-token-at-a-time inference:

- `hidden_state`
- `residual`
- `q_state`, `k_state`, `v_state`
- `k_cache`, `v_cache`
- attention `scores`
- attention `context`
- MLP scratch tensors
- `logits`
- `probs`

The KV cache is important. It stores prior keys and values so each new token can attend to previous tokens without recomputing every previous layer from scratch.

## 6. The model is constructed

The runtime selects a model template based on quantization.

For example:

- f32 uses `Model<float, float>`
- Q8F16 uses int8 matmul weights and f16 auxiliary weights

`Model` is defined in `include/model/modules.h` and implemented in `src/model/mistral/modules.cpp`.

It owns:

- embedding table
- decoder layers
- final norm
- LM head

Each submodule receives tensor views from `Parameters`.

## 7. Prompt prefill warms the cache

Before generating new text, the CLI runs the prompt tokens except the last one:

```cpp
for (int i=0; i<(int)got.size()-1; i++){
    model.forward(infer, got[i]);
}
```

This advances `infer.pos` and fills the KV cache with prompt context.

Then generation starts from the final prompt token.

## 8. One token forward pass

`Model::forward(infer, token_id)` is the core path.

It does:

1. Embedding lookup
2. Run every decoder layer
3. Final RMSNorm
4. LM head projection
5. Increment `infer.pos`

After this call, `infer.logits` contains one score per vocabulary token.

## 9. Embedding

`Embedding::forward()` looks up one row in the embedding table:

```cpp
Tensor<TAux> row = table.at({token_id});
```

It copies that row into `infer.hidden_state`.

At this point the token id has become a dense vector.

## 10. Decoder layer

Each layer follows the usual decoder block shape:

1. Save residual.
2. RMSNorm.
3. Attention.
4. Add residual.
5. Save residual.
6. RMSNorm.
7. MLP.
8. Add residual.

The layer mutates `infer.hidden_state` in place.

## 11. Attention

Attention starts by projecting the hidden state into Q, K, and V:

```cpp
matmul(infer.q_state, q_proj, infer.hidden_state);
matmul(infer.k_state, k_proj, infer.hidden_state);
matmul(infer.v_state, v_proj, infer.hidden_state);
```

Then RoPE is applied to Q and K. RoPE uses the current position `infer.pos` to rotate the query/key vectors so attention knows where the token is in the sequence.

The current K and V are pushed into the KV cache:

```cpp
infer.push_kv(layer);
```

Then each query head attends over cached keys and values from positions `0..infer.pos`.

The flow per head is:

1. Dot Q against cached K.
2. Scale by `1 / sqrt(head_dim)`.
3. Softmax into attention weights.
4. Weighted sum over cached V.
5. Store the result in `infer.context`.

Finally, the output projection writes the attention result back into `infer.hidden_state`.

## 12. MLP

The feed-forward block uses SwiGLU:

1. Project hidden state through `gate_proj`.
2. Apply SiLU to the gate.
3. Project hidden state through `up_proj`.
4. Multiply gate and up elementwise.
5. Project back down with `down_proj`.

The result replaces `infer.hidden_state`, then the layer adds the residual around it.

## 13. LM head produces logits

After all layers, the model applies final RMSNorm and the LM head:

```cpp
matmul(infer.logits, lm_head, infer.hidden_state);
```

`infer.logits` has shape `[vocab_size]`.

Each value is an unnormalized score for one possible next token.

## 14. Sampling picks the next token

`generate()` in `src/main.cpp` calls the model forward pass, applies a repetition penalty, then samples.

If `--temp 0`, it uses greedy decoding:

```cpp
sample_max(infer)
```

That returns the token id with the highest logit.

If temperature is greater than zero, logits are divided by temperature, passed through softmax, and sampled with multinomial sampling.

## 15. Token id becomes text

The sampled id is appended to history, checked against EOS, decoded back into text, and printed:

```cpp
std::cout << params->tokenizer.decode({t}) << std::flush;
```

Then the new token is fed back into `generate()`, using the same `InferenceState` and KV cache.

That loop is text generation: predict one token, print it, append it, repeat.

## Current status

The loader and test path now validate Qwen3 `.mog` files. The full transformer implementation currently lives under `src/model/mistral/`, and architecture-specific Qwen inference work still needs to be added before Qwen generation is complete.
