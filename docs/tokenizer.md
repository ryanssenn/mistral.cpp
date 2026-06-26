# Qwen Tokenizer Flow

The tokenizer converts text into token IDs in four main stages.

```
Input text
    ↓
Split around special tokens
    ↓
Pre-tokenize normal text with regex
    ↓
Byte-level BPE
    ↓
Token IDs
```

## 1. Split around special tokens

Special tokens such as:

```
<|im_start|>
<|im_end|>
```

are preserved as a single token.

Everything between them is treated as normal text.

---

## 2. Pre-tokenization

Normal text is split using Qwen's regex.

The goal is to separate words, punctuation, spaces, numbers, and newlines before BPE.

For example:

```
Hello, world!
```

becomes roughly:

```
"Hello"
","
" world"
"!"
```

Each piece is processed independently.

---

## 3. Byte-level BPE

Each piece is converted into bytes.

Every byte (0-255) is mapped to a visible Unicode character using the fixed GPT-2 byte table.

For example:

```
32 (space)
    ↓
"Ġ"
```

These Unicode strings are looked up in the vocabulary to obtain the initial token IDs.

BPE then repeatedly merges neighboring token IDs according to the learned merge rules until no more merges are possible.

---

## 4. Output

The token IDs from every piece are concatenated together.

Special tokens are inserted directly into the output whenever they appear in the input.

The final result is a sequence of token IDs ready for the model.