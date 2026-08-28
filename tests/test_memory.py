import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "estimate_memory.py"
spec = importlib.util.spec_from_file_location("estimate_memory", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_kv_cache_arithmetic() -> None:
    values = module.estimate(
        context=32768,
        concurrent_sequences=1,
        tensor_parallel=1,
        weight_bits=16,
    )
    assert values["kv_per_token_kib"] == 256
    assert values["kv_total_gib"] == 8
    assert 60 < values["weights_total_gib"] < 62


def test_four_bit_weight_arithmetic() -> None:
    values = module.estimate(
        context=8192,
        concurrent_sequences=1,
        tensor_parallel=1,
        weight_bits=4,
    )
    assert 15 < values["weights_total_gib"] < 16
    assert values["kv_total_gib"] == 2


def test_all_linear_lora_parameter_count_exceeds_attention_only() -> None:
    all_linear = module.lora_parameters(16, target="all-linear")
    attention = module.lora_parameters(16, target="attention")
    assert all_linear > attention > 0
    assert 130_000_000 < all_linear < 140_000_000
