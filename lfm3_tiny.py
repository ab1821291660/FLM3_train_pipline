"""A small, dependency-light LFM3 architecture reproduction for smoke training.

This module keeps the public LFM3 backbone ideas while reducing dimensions so
that every stage can run on CPU.  It is an educational reproduction, not an
implementation whose parameters are compatible with the 1.7B checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TinyLFM3Config:
    vocab_size: int = 259
    hidden_size: int = 64
    intermediate_size: int = 160
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 256
    conv_kernel_size: int = 7
    sliding_window: int = 32
    norm_eps: float = 1e-5
    rope_theta_global: float = 1_000_000.0
    rope_theta_sliding: float = 10_000.0
    global_partial_rotary_factor: float = 0.5
    layer_types: list[str] = field(
        default_factory=lambda: ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
    )

    def validate(self) -> None:
        allowed = {"conv", "full_attention", "sliding_attention"}
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if not set(self.layer_types) <= allowed:
            raise ValueError(f"unsupported layer type: {set(self.layer_types) - allowed}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rotary_dim = int((self.hidden_size // self.num_attention_heads) * self.global_partial_rotary_factor)
        if rotary_dim % 2:
            raise ValueError("the partial rotary dimension must be even")

    def to_dict(self) -> dict:
        return asdict(self)


class ByteTokenizer:
    """UTF-8 byte tokenizer used only by the tiny offline reproduction.

    IDs 0, 1 and 2 are PAD, BOS and EOS.  Every byte maps to byte + 3, so the
    vocabulary has 259 entries and needs no tokenizer download.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids = ([self.bos_token_id] if add_bos else []) + [value + 3 for value in text.encode("utf-8")]
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = self.eos_token_id
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        values = bytes(token - 3 for token in ids if 3 <= int(token) <= 258)
        return values.decode("utf-8", errors="replace")

    def batch(
        self,
        sequences: list[list[int]],
        labels: list[list[int]] | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        label_tensor = torch.full((len(sequences), max_length), -100, dtype=torch.long) if labels else None
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, : len(sequence)] = 1
            if labels is not None and label_tensor is not None:
                label_tensor[index, : len(labels[index])] = torch.tensor(labels[index], dtype=torch.long)
        return input_ids.to(device), attention_mask.to(device), None if label_tensor is None else label_tensor.to(device)


def format_chat(messages: list[dict]) -> str:
    """Render a compact ChatML representation compatible with LFM3 concepts."""

    parts = ["<|startoftext|>"]
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
    return "".join(parts)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return normalized * self.weight


def _apply_rotary(x: Tensor, positions: Tensor, rotary_dim: int, theta: float) -> Tensor:
    if rotary_dim == 0:
        return x
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    frequencies = torch.outer(positions.float(), inv_freq)
    cos = frequencies.cos().repeat_interleave(2, dim=-1)[None, None, :, :].to(x.dtype)
    sin = frequencies.sin().repeat_interleave(2, dim=-1)[None, None, :, :].to(x.dtype)
    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    pairs = rotated.reshape(*rotated.shape[:-1], rotary_dim // 2, 2)
    rotated_half = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return torch.cat((rotated * cos + rotated_half * sin, remainder), dim=-1)


class HeadGatedAttention(nn.Module):
    def __init__(self, config: TinyLFM3Config, layer_type: str):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_type = layer_type
        self.sliding_window = config.sliding_window
        self.rotary_dim = self.head_dim if layer_type == "sliding_attention" else int(
            self.head_dim * config.global_partial_rotary_factor
        )
        self.rope_theta = config.rope_theta_sliding if layer_type == "sliding_attention" else config.rope_theta_global

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        positions = torch.arange(sequence_length, device=hidden_states.device)
        q = _apply_rotary(q, positions, self.rotary_dim, self.rope_theta)
        k = _apply_rotary(k, positions, self.rotary_dim, self.rope_theta)
        repeats = self.num_heads // self.num_key_value_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        row = torch.arange(sequence_length, device=hidden_states.device)[:, None]
        column = torch.arange(sequence_length, device=hidden_states.device)[None, :]
        allowed = column <= row
        if self.layer_type == "sliding_attention":
            allowed = allowed & (column > row - self.sliding_window)
        scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
        context = torch.matmul(probabilities, v)
        head_gate = torch.sigmoid(self.gate_proj(hidden_states)).transpose(1, 2).unsqueeze(-1)
        context = (context * head_gate).transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.o_proj(context)


class GatedShortConv(nn.Module):
    def __init__(self, config: TinyLFM3Config):
        super().__init__()
        self.kernel_size = config.conv_kernel_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.kernel_size,
            groups=config.hidden_size,
            padding=self.kernel_size - 1,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        b_gate, c_gate, value = self.in_proj(hidden_states).transpose(1, 2).chunk(3, dim=1)
        convolved = self.conv(b_gate * value)[..., : hidden_states.shape[1]]
        output = self.out_proj((c_gate * convolved).transpose(1, 2).contiguous())
        return output


class SwiGLU(nn.Module):
    def __init__(self, config: TinyLFM3Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class TinyLFM3Block(nn.Module):
    def __init__(self, config: TinyLFM3Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.operator = GatedShortConv(config) if layer_type == "conv" else HeadGatedAttention(config, layer_type)
        self.feed_forward = SwiGLU(config)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.post_operator_norm(
            self.operator(self.operator_norm(hidden_states), attention_mask)
        )
        hidden_states = hidden_states + self.post_ffn_norm(self.feed_forward(self.ffn_norm(hidden_states)))
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class TinyLFM3ForCausalLM(nn.Module):
    def __init__(self, config: TinyLFM3Config | None = None):
        super().__init__()
        self.config = config or TinyLFM3Config()
        self.config.validate()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.embedding_norm = RMSNorm(self.config.hidden_size, self.config.norm_eps)
        self.layers = nn.ModuleList(
            TinyLFM3Block(self.config, layer_type) for layer_type in self.config.layer_types
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds {self.config.max_position_embeddings}"
            )
        hidden_states = self.embedding_norm(self.embed_tokens(input_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_summary(self) -> dict[str, int]:
        return {
            "parameters": self.parameter_count(),
            "layers": len(self.layers),
            "conv": self.config.layer_types.count("conv"),
            "sliding_attention": self.config.layer_types.count("sliding_attention"),
            "full_attention": self.config.layer_types.count("full_attention"),
        }


__all__ = [
    "ByteTokenizer",
    "TinyLFM3Config",
    "TinyLFM3ForCausalLM",
    "format_chat",
]
