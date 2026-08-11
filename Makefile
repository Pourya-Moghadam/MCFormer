.PHONY: smoke test lint typecheck build check

smoke:
	PYTHONPATH=src python -m compileall -q src tests scripts
	PYTHONPATH=src python -m mcformer.cli.show_config --config configs/experiment/e01_toyota_cs_mcformer.yaml > /dev/null

test:
	PYTHONPATH=src python -m pytest

lint:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts

typecheck:
	python -m mypy src/mcformer

build:
	python -m build

check: smoke lint typecheck test build
