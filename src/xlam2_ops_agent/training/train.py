from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .collator import CompletionOnlyCollator
from .config import QLoRAConfig, load_qlora_config
from .data import build_tokenized_dataset, load_jsonl, read_template
from .manifest import write_training_manifest
from .model import build_quantized_lora_model, load_tokenizer
from .preflight import collect_preflight


def _filter_supported(callable_object: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _training_arguments(config: QLoRAConfig, *, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    settings = config.training
    kwargs: dict[str, Any] = {
        "output_dir": str(settings.output_dir),
        "seed": settings.seed,
        "data_seed": settings.seed,
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "per_device_eval_batch_size": settings.per_device_eval_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "num_train_epochs": settings.num_train_epochs,
        "max_steps": settings.max_steps,
        "warmup_steps": settings.warmup_steps,
        "lr_scheduler_type": settings.lr_scheduler_type,
        "logging_strategy": "steps",
        "logging_steps": settings.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": settings.eval_steps if has_eval else None,
        "save_strategy": "steps",
        "save_steps": settings.save_steps,
        "save_total_limit": settings.save_total_limit,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": settings.gradient_checkpointing_use_reentrant
        },
        "optim": settings.optim,
        "bf16": settings.bf16,
        "tf32": settings.tf32,
        "dataloader_num_workers": settings.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "train_sampling_strategy": "group_by_length",
        "length_column_name": "length",
        "report_to": settings.report_to,
        "run_name": settings.output_dir.name,
        "logging_dir": str(settings.output_dir / "tensorboard"),
        "torch_empty_cache_steps": settings.torch_empty_cache_steps,
        "prediction_loss_only": True,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    supported = _filter_supported(TrainingArguments.__init__, kwargs)
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        raise RuntimeError(
            "Pinned Transformers no longer accepts required TrainingArguments: "
            + ", ".join(dropped)
        )
    return TrainingArguments(**supported)


def run_training(config: QLoRAConfig, *, resume_from_checkpoint: str | None = None) -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import Trainer, set_seed

    preflight = collect_preflight(path=config.training.output_dir.parent)
    if preflight["errors"]:
        raise RuntimeError("Preflight failed:\n" + "\n".join(preflight["errors"]))
    torch.backends.cuda.matmul.allow_tf32 = config.training.tf32
    torch.backends.cudnn.allow_tf32 = config.training.tf32
    set_seed(config.training.seed)

    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    (config.training.output_dir / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config.training.output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8"
    )

    tokenizer = load_tokenizer(config)
    chat_template = read_template(config.data.template_file)
    train_dataset, train_stats = build_tokenized_dataset(
        tokenizer,
        rows=load_jsonl(config.data.train_file),
        chat_template=chat_template,
        max_seq_length=config.data.max_seq_length,
        num_proc=config.data.preprocessing_num_workers,
        overwrite_cache=config.data.overwrite_cache,
    )
    eval_dataset = None
    eval_stats = None
    if config.data.eval_file is not None:
        eval_dataset, eval_stats = build_tokenized_dataset(
            tokenizer,
            rows=load_jsonl(config.data.eval_file),
            chat_template=chat_template,
            max_seq_length=config.data.max_seq_length,
            num_proc=config.data.preprocessing_num_workers,
            overwrite_cache=config.data.overwrite_cache,
        )

    model = build_quantized_lora_model(config)
    collator = CompletionOnlyCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=config.training.pad_to_multiple_of,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_arguments(config, has_eval=eval_dataset is not None),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "processing_class": tokenizer,
    }
    supported_trainer_kwargs = _filter_supported(Trainer.__init__, trainer_kwargs)
    dropped = sorted(set(trainer_kwargs) - set(supported_trainer_kwargs))
    if dropped:
        raise RuntimeError(
            "Pinned Transformers no longer accepts required Trainer parameters: "
            + ", ".join(dropped)
        )
    trainer = Trainer(**supported_trainer_kwargs)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    final_adapter = config.training.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)

    train_metrics = dict(result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics: dict[str, Any] | None = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    stats = {
        "train": train_stats.as_dict(),
        "eval": eval_stats.as_dict() if eval_stats else None,
        "metrics": {"train": train_metrics, "eval": eval_metrics},
    }
    write_training_manifest(
        config=config,
        output_path=config.training.output_dir / "training_manifest.json",
        data_stats=stats,
        model=model,
    )
    return final_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune xLAM-2-32B with single-GPU QLoRA")
    parser.add_argument("--config", default="configs/a100_40gb_qlora.yaml")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, help="Override max_steps from YAML")
    parser.add_argument("--output-dir", help="Override output_dir from YAML")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and report the trajectory files without loading the model",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_qlora_config(args.config)
    if args.max_steps is not None:
        config = config.model_copy(
            update={
                "training": config.training.model_copy(update={"max_steps": args.max_steps})
            }
        )
    if args.output_dir:
        config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={"output_dir": Path(args.output_dir).resolve()}
                )
            }
        )

    if args.validate_only:
        from .data import validate_trajectories

        train_rows, train_stats = validate_trajectories(load_jsonl(config.data.train_file))
        payload: dict[str, Any] = {
            "train": train_stats.as_dict(),
            "train_ids": [row["id"] for row in train_rows],
        }
        if config.data.eval_file:
            eval_rows, eval_stats = validate_trajectories(load_jsonl(config.data.eval_file))
            payload["eval"] = eval_stats.as_dict()
            payload["eval_ids"] = [row["id"] for row in eval_rows]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    final_adapter = run_training(config, resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"Final adapter saved to {final_adapter}")


if __name__ == "__main__":
    main()
