import pytest

torch = pytest.importorskip("torch")

from xlam2_ops_agent.training.collator import CompletionOnlyCollator


def test_completion_only_collator_right_pads() -> None:
    collator = CompletionOnlyCollator(pad_token_id=99, pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
    )
    assert tuple(batch["input_ids"].shape) == (2, 4)
    assert batch["input_ids"][1].tolist() == [4, 99, 99, 99]
    assert batch["attention_mask"][1].tolist() == [1, 0, 0, 0]
    assert batch["labels"][1].tolist() == [4, -100, -100, -100]
