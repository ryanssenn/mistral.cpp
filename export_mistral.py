import os
import json
import struct

import safetensors
import torch
import argparse

"""
Usage:
  python export_mistral.py --model_dir /path/to/Mistral-7B-v0.1 [--out mistral.mog] [--quant f32]

Converts a Hugging Face Mistral checkpoint into a single .mog file.
Default (--quant Q8F16): int8 gate/up, f16 for all other weights.
"""

MAGIC = b"MOG\x00"
FORMAT_VERSION = 1

KV_STRING = 0
KV_UINT32 = 1
KV_FLOAT32 = 2

DTYPE_F32 = 0
DTYPE_INT8 = 1
DTYPE_F16 = 2

DTYPE_BYTES = {
    DTYPE_F32: 4,
    DTYPE_INT8: 1,
    DTYPE_F16: 2,
}


def write_u8(buf, v):
    buf.extend(struct.pack("<B", v))


def write_u32(buf, v):
    buf.extend(struct.pack("<I", v))


def write_u64(buf, v):
    buf.extend(struct.pack("<Q", v))


def write_f32(buf, v):
    buf.extend(struct.pack("<f", v))


def write_string(buf, s):
    b = s.encode("utf-8")
    write_u32(buf, len(b))
    buf.extend(b)


def write_kv_uint32(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_UINT32)
    write_u32(buf, value)


def write_kv_float32(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_FLOAT32)
    write_f32(buf, float(value))


def write_kv_string(buf, key, value):
    write_string(buf, key)
    write_u8(buf, KV_STRING)
    write_string(buf, value)


def pad_to_64(offset):
    r = offset % 64
    return 0 if r == 0 else 64 - r


def quantize(x: torch.Tensor, n_bits: int, group_size: int):
    assert x.numel() % group_size == 0
    x = x.reshape(-1, group_size).float()
    int_max = 2 ** (n_bits - 1) - 1
    scales = int_max / x.abs().max(dim=-1).values.unsqueeze(-1)
    quant = (x * scales).round().clamp(-int_max, int_max)
    return quant, scales


def should_quantize(quant_mode, tensor_name):
    return quant_mode != "f32" and any(
        key in tensor_name for key in ("mlp.gate_proj", "mlp.up_proj")
    )

def load_config(model_dir, quant_mode):
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    return {
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "vocab_size": cfg["vocab_size"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "sliding_window": cfg["sliding_window"] if cfg["sliding_window"] is not None else 0,
        "rope_theta": cfg["rope_theta"],
        "norm_eps": cfg["rms_norm_eps"],
        "quant": quant_mode,
    }


def load_tokenizer(model_dir):
    with open(os.path.join(model_dir, "tokenizer.json")) as fh:
        t = json.load(fh)
    return t["model"]["vocab"], t["model"]["merges"]


def load_tensor_map(model_dir, weight_map, quant_mode, data_size, group_size):
    tensors = {}
    start = 0

    for tensor_name in weight_map:
        tensor_file_path = os.path.join(model_dir, weight_map[tensor_name])
        with safetensors.safe_open(tensor_file_path, framework="pt") as handle:
            tensor = handle.get_tensor(tensor_name)

            if should_quantize(quant_mode, tensor_name):
                tensors[tensor_name] = {
                    "dtype": DTYPE_INT8,
                    "shape": list(tensor.shape)[:4],
                    "offset": start,
                }
                start += tensor.numel() * data_size
                start += pad_to_64(start)

                scale_size = tensor.numel() // group_size
                tensors[tensor_name]["scale_offset"] = start
                tensors[tensor_name]["scale_size"] = scale_size
                start += scale_size * 4
                start += pad_to_64(start)
            else:
                dtype = DTYPE_F32 if quant_mode == "f32" else DTYPE_F16
                tensors[tensor_name] = {
                    "dtype": dtype,
                    "shape": list(tensor.shape)[:4],
                    "offset": start,
                    "scale_offset": 0,
                    "scale_size": 0,
                }
                start += tensor.numel() * DTYPE_BYTES[dtype]
                start += pad_to_64(start)

    return tensors, start


def build_header_blob(config, vocab, merges, tensors, weight_map):
    buf = bytearray()

    write_string(buf, "mistral")
    write_u32(buf, 11)
    write_kv_uint32(buf, "hidden_size", config["hidden_size"])
    write_kv_uint32(buf, "intermediate_size", config["intermediate_size"])
    write_kv_uint32(buf, "n_layers", config["n_layers"])
    write_kv_uint32(buf, "n_heads", config["n_heads"])
    write_kv_uint32(buf, "n_kv_heads", config["n_kv_heads"])
    write_kv_uint32(buf, "vocab_size", config["vocab_size"])
    write_kv_uint32(buf, "sliding_window", config["sliding_window"])
    write_kv_uint32(buf, "max_position_embeddings", config["max_position_embeddings"])
    write_kv_float32(buf, "rope_theta", config["rope_theta"])
    write_kv_float32(buf, "norm_eps", config["norm_eps"])
    write_kv_string(buf, "quant", config["quant"])

    write_u32(buf, len(vocab))
    for token, token_id in vocab.items():
        write_string(buf, token)
        write_u32(buf, int(token_id))

    write_u32(buf, len(merges))
    for merge in merges:
        write_string(buf, merge)

    write_u32(buf, len(tensors))
    for tensor_name in weight_map:
        info = tensors[tensor_name]
        shape = info["shape"]
        ndim = len(shape)
        write_string(buf, tensor_name)
        write_u8(buf, info["dtype"])
        write_u8(buf, ndim)
        for d in range(4):
            write_u32(buf, shape[d] if d < ndim else 0)
        write_u64(buf, info["offset"])
        write_u64(buf, info.get("scale_offset", 0))
        write_u32(buf, info.get("scale_size", 0))

    return bytes(buf)


def write_tensor(out, tensor, base_offset, tensor_offset):
    out.seek(base_offset + tensor_offset, 0)
    out.write(tensor.numpy().tobytes())


def write_binary(model_dir, out_path, weight_map, tensors, header_blob, quant_mode, data_size, data_type, group_size):
    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", FORMAT_VERSION))
        out.write(struct.pack("<Q", len(header_blob)))
        out.write(header_blob)

        out.seek(pad_to_64(out.tell()), 1)
        base_offset = out.tell()

        for tensor_name in weight_map:
            tensor_file_path = os.path.join(model_dir, weight_map[tensor_name])
            with safetensors.safe_open(tensor_file_path, framework="pt") as handle:
                tensor = handle.get_tensor(tensor_name)
                scales = None

                if should_quantize(quant_mode, tensor_name):
                    tensor, scales = quantize(tensor, data_size * 8, group_size)
                    tensor = tensor.to(data_type)
                    scales = scales.to(torch.float32)
                elif quant_mode == "f32":
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(torch.float16)

                write_tensor(out, tensor, base_offset, tensors[tensor_name]["offset"])
                if scales is not None:
                    write_tensor(out, scales, base_offset, tensors[tensor_name]["scale_offset"])

        payload_size = out.tell() - base_offset

    expected_payload = max(
        info["offset"] + (
            int(torch.prod(torch.tensor(info["shape"])).item()) * DTYPE_BYTES[info["dtype"]]
        )
        for info in tensors.values()
    )
    if payload_size < expected_payload:
        raise ValueError(f"payload too small: wrote {payload_size}, need at least {expected_payload}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out", default="./mistral.mog")
    parser.add_argument("--quant", default="Q8F16", choices=["f32", "Q8F16"])
    args = parser.parse_args()

    data_size = 1 if args.quant != "f32" else 4
    data_type = torch.int8 if args.quant != "f32" else torch.float32
    group_size = 64

    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]

    print(f"Exporting {args.model_dir} -> {args.out} ({args.quant})")

    config = load_config(args.model_dir, args.quant)
    vocab, merges = load_tokenizer(args.model_dir)
    tensors, _ = load_tensor_map(
        args.model_dir, weight_map, args.quant, data_size, group_size
    )
    header_blob = build_header_blob(config, vocab, merges, tensors, weight_map)
    write_binary(
        args.model_dir,
        args.out,
        weight_map,
        tensors,
        header_blob,
        args.quant,
        data_size,
        data_type,
        group_size,
    )

    print("Completed")


if __name__ == "__main__":
    main()
