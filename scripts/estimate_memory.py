#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class XlamSpec:
    parameters: int = 32_763_900_000
    layers: int = 64
    hidden_size: int = 5120
    intermediate_size: int = 27648
    attention_heads: int = 40
    kv_heads: int = 8

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attention_heads

    @property
    def kv_projection_size(self) -> int:
        return self.kv_heads * self.head_dim


def gib(value: int | float) -> float:
    return float(value) / (1024**3)


def lora_parameters(
    rank: int,
    *,
    target: Literal["all-linear", "attention"] = "all-linear",
    spec: XlamSpec | None = None,
) -> int:
    if rank <= 0:
        raise ValueError("rank must be positive")
    model = spec or XlamSpec()
    hidden = model.hidden_size
    kv = model.kv_projection_size
    per_layer = rank * (
        (hidden + hidden)  # q_proj
        + (hidden + kv)  # k_proj
        + (hidden + kv)  # v_proj
        + (hidden + hidden)  # o_proj
    )
    if target == "all-linear":
        per_layer += rank * (
            (hidden + model.intermediate_size)  # gate_proj
            + (hidden + model.intermediate_size)  # up_proj
            + (model.intermediate_size + hidden)  # down_proj
        )
    return per_layer * model.layers


def estimate(
    *,
    context: int,
    concurrent_sequences: int,
    tensor_parallel: int,
    weight_bits: int = 4,
    kv_bits: int = 16,
    lora_rank: int = 16,
    lora_target: Literal["all-linear", "attention"] = "all-linear",
) -> dict[str, float]:
    if min(context, concurrent_sequences, tensor_parallel, weight_bits, kv_bits, lora_rank) <= 0:
        raise ValueError("all numeric inputs must be positive")
    spec = XlamSpec()
    weight_bytes = spec.parameters * weight_bits / 8
    kv_bytes_per_token = 2 * spec.layers * spec.kv_heads * spec.head_dim * (kv_bits / 8)
    kv_bytes = kv_bytes_per_token * context * concurrent_sequences
    adapter_parameters = lora_parameters(lora_rank, target=lora_target, spec=spec)
    adapter_bf16_bytes = adapter_parameters * 2
    return {
        "weights_total_gib": gib(weight_bytes),
        "kv_per_token_kib": kv_bytes_per_token / 1024,
        "kv_total_gib": gib(kv_bytes),
        "lora_trainable_parameters_millions": adapter_parameters / 1_000_000,
        "lora_bf16_weights_gib": gib(adapter_bf16_bytes),
        "ideal_weights_per_gpu_gib": gib(weight_bytes) / tensor_parallel,
        "ideal_kv_per_gpu_gib": gib(kv_bytes) / tensor_parallel,
        "ideal_weight_plus_kv_per_gpu_gib": gib(weight_bytes + kv_bytes) / tensor_parallel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idealized xLAM-2-32B weight, KV-cache, and LoRA arithmetic."
    )
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--concurrent-sequences", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--kv-bits", type=int, choices=(8, 16), default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--lora-target",
        choices=("all-linear", "attention"),
        default="all-linear",
    )
    args = parser.parse_args()

    values = estimate(
        context=args.context,
        concurrent_sequences=args.concurrent_sequences,
        tensor_parallel=args.tensor_parallel,
        weight_bits=args.weight_bits,
        kv_bits=args.kv_bits,
        lora_rank=args.lora_rank,
        lora_target=args.lora_target,
    )
    print(
        "Idealized estimate only; excludes quantization metadata, activations, gradients, "
        "optimizer state, allocator reserve, CUDA graphs, fragmentation, and runtime buffers."
    )
    for key, value in values.items():
        print(f"{key:42s} {value:12.3f}")


if __name__ == "__main__":
    main()
