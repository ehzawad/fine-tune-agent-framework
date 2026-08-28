from pathlib import Path

from xlam2_ops_agent.training.preflight import _existing_ancestor


def test_existing_ancestor_accepts_future_output_directory(tmp_path: Path) -> None:
    future = tmp_path / "runs" / "experiment" / "checkpoints"
    assert _existing_ancestor(future) == tmp_path
