from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8
    label_pad_token_id: int = -100

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for training") from exc

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of > 1:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            length = len(feature["input_ids"])
            if len(feature["labels"]) != length or len(feature["attention_mask"]) != length:
                raise ValueError("input_ids, labels, and attention_mask lengths must match")
            padding = max_length - length
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            attention_masks.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [self.label_pad_token_id] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
