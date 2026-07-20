.PHONY: all data phase1 test lint fmt

# stdlib-only pull; runs with any python3, no env needed
data:
	python3 scripts/download_raw.py

phase1:
	uv run python -m src.run_phase1

phase2:
	uv run python -m src.run_phase2

phase3:
	uv run python -m src.run_phase3

phase4:
	uv run python -m src.run_phase4

phase5:
	uv run python -m src.run_phase5

phase6:
	uv run python -m src.run_phase6

# Phase 7 evaluates the final holdout. It exists to be run once (and re-run
# only to REPRODUCE that single evaluation - never to iterate).
phase7:
	uv run python -m src.run_phase7

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run mypy src tests scripts

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

# reproduces every number and figure in the README (data must be pulled first)
all: phase1 phase2 phase3 phase4 phase5 phase6 phase7
