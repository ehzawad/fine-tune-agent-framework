from __future__ import annotations

import os
from typing import Any

from .config import QLoRAConfig


def _torch_dtype(name: str) -> Any:
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_tokenizer(config: QLoRAConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with .[train]") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quantized_lora_model(config: QLoRAConfig) -> Any:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with .[train]") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for xLAM-2-32B QLoRA training")
    if config.training.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU/runtime does not report bfloat16 support")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    compute_dtype = _torch_dtype(config.quantization.compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": local_rank},
        attn_implementation=config.model.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": config.training.gradient_checkpointing_use_reentrant
        },
    )
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    if trainable >= total:
        raise RuntimeError("The base model was not frozen; refusing full-model training")
    model.print_trainable_parameters()
    return model
