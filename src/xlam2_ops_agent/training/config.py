from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(StrictConfig):
    model_id: str = "Salesforce/xLAM-2-32b-fc-r"
    revision: str | None = "5ddef330ce01999a05ff56726c543bd6a5fe7142"
    trust_remote_code: bool = False
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"


class QuantizationSettings(StrictConfig):
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"

    @model_validator(mode="after")
    def require_four_bit(self) -> "QuantizationSettings":
        if not self.load_in_4bit:
            raise ValueError("This single-A100 recipe intentionally requires 4-bit loading")
        return self


class LoraSettings(StrictConfig):
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"

    @model_validator(mode="after")
    def validate_targets(self) -> "LoraSettings":
        if isinstance(self.target_modules, list) and not self.target_modules:
            raise ValueError("target_modules cannot be an empty list")
        return self


class DataSettings(StrictConfig):
    train_file: Path
    eval_file: Path | None = None
    template_file: Path = Path("templates/tool_chat_template_xlam_qwen.jinja")
    max_seq_length: int = Field(default=2048, ge=256, le=32768)
    preprocessing_num_workers: int = Field(default=1, ge=1)
    overwrite_cache: bool = False


class TrainingSettings(StrictConfig):
    output_dir: Path = Path("runs/xlam2-a100-qlora")
    seed: int = 42
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = -1
    # Transformers 5.x accepts either an integer step count or a float in [0, 1)
    # in warmup_steps. A float is interpreted as a ratio of total training steps.
    warmup_steps: float = Field(default=0.03, ge=0.0)
    lr_scheduler_type: str = "cosine"
    logging_steps: int = Field(default=1, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=25, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = Field(default=0, ge=0)
    pad_to_multiple_of: int = Field(default=8, ge=1)
    torch_empty_cache_steps: int | None = Field(default=10, ge=1)
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])

    @model_validator(mode="after")
    def validate_training_settings(self) -> "TrainingSettings":
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.warmup_steps >= 1 and not float(self.warmup_steps).is_integer():
            raise ValueError(
                "warmup_steps must be an integer count or a float ratio in [0, 1)"
            )
        return self


class QLoRAConfig(StrictConfig):
    model: ModelSettings = Field(default_factory=ModelSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    data: DataSettings
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    @model_validator(mode="after")
    def validate_precision(self) -> "QLoRAConfig":
        if self.training.bf16 and self.quantization.compute_dtype != "bfloat16":
            raise ValueError(
                "training.bf16=true requires quantization.compute_dtype=bfloat16"
            )
        return self


def _project_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    return parent.parent if parent.name == "configs" else parent


def _resolve_relative_paths(config: QLoRAConfig, config_path: Path) -> QLoRAConfig:
    root = _project_root(config_path)

    def resolve(path: Path | None) -> Path | None:
        if path is None or path.is_absolute():
            return path
        return (root / path).resolve()

    data = config.data.model_copy(
        update={
            "train_file": resolve(config.data.train_file),
            "eval_file": resolve(config.data.eval_file),
            "template_file": resolve(config.data.template_file),
        }
    )
    training = config.training.model_copy(
        update={"output_dir": resolve(config.training.output_dir)}
    )
    return config.model_copy(update={"data": data, "training": training})


def load_qlora_config(path: str | Path) -> QLoRAConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    config = QLoRAConfig.model_validate(raw)
    return _resolve_relative_paths(config, config_path)
