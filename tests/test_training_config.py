from pathlib import Path

import pytest
from pydantic import ValidationError

from xlam2_ops_agent.training.config import QLoRAConfig, load_qlora_config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_resolves_project_paths() -> None:
    config = load_qlora_config(ROOT / "configs" / "a100_40gb_smoke.yaml")
    assert config.data.train_file == (ROOT / "data" / "demo_train.jsonl").resolve()
    assert config.data.template_file == (
        ROOT / "templates" / "tool_chat_template_xlam_qwen.jinja"
    ).resolve()
    assert config.training.output_dir == (ROOT / "runs" / "xlam2-a100-smoke").resolve()
    assert config.training.warmup_steps == 0.0


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate(
            {
                "data": {"train_file": "train.jsonl"},
                "training": {"mystery_knob": True},
            }
        )


def test_bf16_training_requires_bf16_compute() -> None:
    with pytest.raises(ValidationError, match="compute_dtype=bfloat16"):
        QLoRAConfig.model_validate(
            {
                "quantization": {"compute_dtype": "float16"},
                "data": {"train_file": "train.jsonl"},
                "training": {"bf16": True},
            }
        )


def test_warmup_rejects_fractional_step_counts() -> None:
    with pytest.raises(ValidationError, match="integer count"):
        QLoRAConfig.model_validate(
            {
                "data": {"train_file": "train.jsonl"},
                "training": {"warmup_steps": 2.5},
            }
        )
