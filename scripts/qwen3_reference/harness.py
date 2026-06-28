"""
Load Qwen3 weights and call individual modules for golden generation.

Each run_* method calls the vendored transformers forward unchanged and returns
the output tensor(s). Use dump() to write flat f32 lines for C++ unit tests.

Example:
  from qwen3_reference.harness import load, dump

  ref = load(config_path, weights_path)
  hidden = ref.embedding(token_id=151643)
  dump("build/goldens/embedding.txt", "embedding", hidden)

  cos, sin = ref.rotary_emb(hidden, pos=0)
  dump("build/goldens/rope.txt", "cos", cos)
  dump("build/goldens/rope.txt", "sin", sin, append=True)
"""

from __future__ import annotations

from pathlib import Path

import torch

from .configuration_qwen3 import Qwen3Config
from .modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ._stubs import DynamicCache

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2].parent / "Qwen3-0.6B-reference/checkpoint/config.json"
)
DEFAULT_HF_MODEL = Path(__file__).resolve().parents[2].parent / "Qwen3-0.6B"


def load(config_path: str | Path, weights: str | Path, device: str = "cpu", dtype=torch.float32) -> "Reference":
    config = Qwen3Config.from_json_file(config_path)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    state = torch.load if str(weights).endswith(".bin") else _load_safetensors
    sd = state(weights)
    model.load_state_dict(sd, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return Reference(model)


def _load_safetensors(path: str | Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def dump(path: str | Path, name: str, tensor: torch.Tensor, *, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = tensor.detach().float().reshape(-1).tolist()
    mode = "a" if append else "w"
    with path.open(mode) as f:
        if not append:
            f.write(f"# {path.name}\n")
        f.write(f"{name}\n")
        f.write(" ".join(str(x) for x in flat) + "\n")


def dump_pair(path: str | Path, entries: dict[str, torch.Tensor], *, append: bool = False) -> None:
    """Write multiple named tensors to one golden file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode) as f:
        if not append:
            f.write(f"# {path.name}\n")
        for name, tensor in entries.items():
            flat = tensor.detach().float().reshape(-1).tolist()
            f.write(f"{name}\n")
            f.write(" ".join(str(x) for x in flat) + "\n")


def transformers_forward(ref: "Reference", token_id: int, pos: int) -> torch.Tensor:
    """Vendored Qwen3ForCausalLM.forward for a single token."""
    input_ids = torch.tensor([[token_id]], device=ref.device)
    position_ids = ref._position_ids(pos)
    cache_position = ref._cache_position(pos)
    with torch.no_grad():
        out = ref.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        )
    return out.logits


def hf_forward(
    model_path: str | Path,
    token_id: int,
    pos: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Installed HuggingFace transformers forward for a single token."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    input_ids = torch.tensor([[token_id]], device=device)
    position_ids = torch.tensor([[pos]], device=device)
    cache_position = torch.tensor([pos], device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        )
    return out.logits


def assert_allclose(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-5) -> float:
    max_err = (a.detach().float() - b.detach().float()).abs().max().item()
    if max_err > atol:
        raise AssertionError(f"{name}: max error {max_err} exceeds atol {atol}")
    return max_err


def _batch_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """[hidden_size] or [1, hidden_size] → [1, 1, hidden_size]."""
    if hidden.dim() == 1:
        return hidden.reshape(1, 1, -1)
    if hidden.dim() == 2:
        return hidden.unsqueeze(1)
    return hidden


class Reference:
    """Holds a loaded Qwen3ForCausalLM; run_* methods map to qmog.cpp modules."""

    def __init__(self, model: Qwen3ForCausalLM) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @property
    def config(self) -> Qwen3Config:
        return self.model.config

    @property
    def backbone(self):
        return self.model.model

    def _position_ids(self, pos: int) -> torch.Tensor:
        return torch.tensor([[pos]], device=self.device)

    def _cache_position(self, pos: int) -> torch.Tensor:
        return torch.tensor([pos], device=self.device)

    # C++: Embedding::forward
    def embedding(self, token_id: int) -> torch.Tensor:
        input_ids = torch.tensor([[token_id]], device=self.device)
        with torch.no_grad():
            return self.backbone.embed_tokens(input_ids)

    # C++: RMSNorm::forward
    def rms_norm(self, module: Qwen3RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return module(_batch_hidden(hidden))

    # C++: RotaryEmbedding::forward
    def rotary_emb(self, hidden: torch.Tensor, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.backbone.rotary_emb(_batch_hidden(hidden), self._position_ids(pos))

    # C++: rope() kernel — apply_rotary_pos_emb on projected q/k
    def rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        with torch.no_grad():
            return apply_rotary_pos_emb(q, k, cos, sin)

    # C++: Attention::forward (full module)
    def attention(
        self,
        layer: int,
        hidden: torch.Tensor,
        pos: int,
        past_key_values: DynamicCache | None = None,
    ) -> torch.Tensor:
        attn = self.backbone.layers[layer].self_attn
        hidden = _batch_hidden(hidden)
        position_embeddings = self.rotary_emb(hidden, pos)
        with torch.no_grad():
            out, _ = attn(
                hidden_states=hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=past_key_values,
                cache_position=self._cache_position(pos),
            )
        return out

    # C++: eager matmul + softmax path inside Attention
    def attention_scores(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask=None,
    ):
        attn = self.backbone.layers[layer].self_attn
        with torch.no_grad():
            return eager_attention_forward(
                attn,
                q,
                k,
                v,
                attention_mask,
                scaling=attn.scaling,
            )

    # C++: MLP::forward
    def mlp(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        module: Qwen3MLP = self.backbone.layers[layer].mlp
        with torch.no_grad():
            return module(_batch_hidden(hidden))

    # C++: Layer::forward
    def layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        pos: int,
        past_key_values: DynamicCache | None = None,
    ) -> torch.Tensor:
        block = self.backbone.layers[layer]
        hidden = _batch_hidden(hidden)
        position_embeddings = self.rotary_emb(hidden, pos)
        mask = None
        with torch.no_grad():
            return block(
                hidden,
                attention_mask=mask,
                position_ids=self._position_ids(pos),
                past_key_values=past_key_values,
                cache_position=self._cache_position(pos),
                position_embeddings=position_embeddings,
            )

    # C++: final RMSNorm in Model::forward
    def norm(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone.norm(_batch_hidden(hidden))

    # C++: LMHead::forward
    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = _batch_hidden(hidden)
        with torch.no_grad():
            if self.config.tie_word_embeddings:
                return torch.nn.functional.linear(hidden, self.backbone.embed_tokens.weight)
            return self.model.lm_head(hidden)

    # C++: Model::forward (one token)
    def forward_token(self, token_id: int, pos: int, past_key_values: DynamicCache | None = None) -> torch.Tensor:
        hidden = self.embedding(token_id)
        position_embeddings = self.rotary_emb(hidden, pos)
        mask = None
        with torch.no_grad():
            out = self.backbone(
                inputs_embeds=hidden,
                attention_mask=mask,
                position_ids=self._position_ids(pos),
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
                cache_position=self._cache_position(pos),
            )
            logits = self.lm_head(out.last_hidden_state)
        return logits

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    def layer_module(self, layer: int):
        return self.backbone.layers[layer]


def module_paths() -> dict[str, str]:
    """Python module → C++ struct (for test naming)."""
    return {
        "Qwen3RMSNorm": "RMSNorm  include/model/model.h",
        "Qwen3RotaryEmbedding": "RotaryEmbedding  include/model/model.h",
        "embed_tokens": "Embedding  include/model/model.h",
        "Qwen3Attention": "Attention  include/model/model.h",
        "Qwen3MLP": "MLP  include/model/model.h",
        "Qwen3DecoderLayer": "Layer  include/model/model.h",
        "norm": "RMSNorm (final)  include/model/model.h",
        "lm_head": "LMHead  include/model/model.h",
        "Qwen3Model": "Model  include/model/model.h",
        "apply_rotary_pos_emb": "rope()  include/backend/kernels.h",
        "eager_attention_forward": "Attention matmul/softmax  src/model/model.cpp",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--list-modules", action="store_true")
    args = parser.parse_args()

    if args.list_modules:
        for py, cpp in module_paths().items():
            print(f"{py:30} {cpp}")
        raise SystemExit(0)

    if args.config is None or args.weights is None:
        parser.error("--config and --weights are required unless --list-modules")

    ref = load(args.config, args.weights)
    print(f"loaded  layers={ref.config.num_hidden_layers}  hidden={ref.config.hidden_size}")
