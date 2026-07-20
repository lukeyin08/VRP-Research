.PHONY: all data phase1 test lint fmt

# stdlib-only pull; runs with any python3, no env needed
data:
	python3 scripts/download_raw.py

phase1:
	uv run python -m src.run_phase1

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run mypy src tests scripts

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

# extended by later phases: phase2 .. phase7, then `all` reproduces everything
all: phase1
