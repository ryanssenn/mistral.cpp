"""Minimal stand-ins for transformers imports used by modeling_qwen3.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar("T")


def _identity(obj: T) -> T:
    return obj


def _factory(*_dec_args, **_dec_kwargs):
    def wrap(obj: T) -> T:
        return obj

    return wrap


auto_docstring = _identity
check_model_inputs = _identity
can_return_tuple = _identity
dynamic_rope_update = _identity
deprecate_kwarg = _factory
use_kernel_forward_from_hub = _factory


class TransformersKwargs(dict):
    pass


class Unpack:
    @classmethod
    def __class_getitem__(cls, _item):
        return Any


class GenerationMixin:
    pass


class GradientCheckpointingLayer(nn.Module):
    pass


class PreTrainedModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def post_init(self) -> None:
        pass


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


@dataclass
class CausalLMOutputWithPast:
    loss: Any = None
    logits: torch.Tensor | None = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


ACT2FN = {"silu": F.silu}


def default_rope_init(config, device=None):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim)
    )
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS = {"default": default_rope_init}


class DynamicCache:
    def __init__(self, config=None) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


Cache = DynamicCache


def create_causal_mask(**_kwargs) -> None:
    return None


def create_sliding_window_causal_mask(**_kwargs) -> None:
    return None


ALL_ATTENTION_FUNCTIONS: dict[str, Callable] = {}


class PretrainedConfig:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def layer_type_validation(layer_types, num_hidden_layers) -> None:
    if len(layer_types) != num_hidden_layers:
        raise ValueError("layer_types length must match num_hidden_layers")


def rope_config_validation(_config) -> None:
    pass


class GenericForSequenceClassification(PreTrainedModel):
    pass


class GenericForTokenClassification(PreTrainedModel):
    pass


class GenericForQuestionAnswering(PreTrainedModel):
    pass
