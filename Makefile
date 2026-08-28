.PHONY: init-env install test lint format demo validate-data preflight train-smoke train
.PHONY: serve serve-adapter smoke-http verify memory zip

init-env:
	./scripts/init_env.zsh

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

demo:
	xlam2-agent demo

validate-data:
	python -m xlam2_ops_agent.training.train --config configs/a100_40gb_qlora.yaml --validate-only

preflight:
	./scripts/preflight_a100.zsh

train-smoke:
	./scripts/smoke_train.zsh

train:
	./scripts/train_qlora.zsh

serve:
	./deployment/serve_vllm.zsh

serve-adapter:
	./deployment/serve_adapter_vllm.zsh

smoke-http:
	python scripts/smoke_http.py --model $${XLAM_MODEL:-xlam-2-32b-fc-r} --api-key $${VLLM_API_KEY:-local-token}

verify:
	python scripts/verify_official_metadata.py

memory:
	python scripts/estimate_memory.py --context 8192 --tensor-parallel 1 --weight-bits 4

zip:
	git archive --format=zip --output=fine-tune-agent-framework.zip HEAD
