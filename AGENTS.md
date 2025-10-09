# Repository Guidelines

## Project Structure & Module Organization
Core reinforcement-learning components live under `memory_rl/`: algorithms, buffers, environments, networks, loggers, and utilities. PopGym baselines and wrappers are grouped in `popjaxrl/`. Tests reside in `tests/`; run-time artifacts and checkpoints accumulate in `outputs/` (timestamped) and visualizations in `plots/`. Packaging metadata lives in `pyproject.toml` and `uv.lock`; hyperparameter sweeps are defined in `memory_rl/sweep.yaml`.

## Build, Test, and Development Commands
- `uv sync` installs the pinned Python ≥3.11 toolchain; add `--extra cuda` when targeting GPU builds.
- `uv run main algorithm=dqn environment=cartpole` launches a reference training job using Hydra overrides.
- `uv run sweep --project memory-rl` coordinates Optuna/W&B sweeps defined in `memory_rl/sweep.yaml`.
- `uv run pytest` executes the test suite; add `-k algorithm` to focus on specific blocks.
- `uv run pre-commit run --all-files` enforces formatters and linters before pushing.

## Coding Style & Naming Conventions
Follow Black (line length 88) and 4-space indentation; keep modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Sorting imports via isort’s Black profile is automatic; resist manual reordering. Flake8 (with Bugbear and comprehensions), Ruff, and MyPy run through pre-commit—address warnings instead of silencing them. Use type hints liberally for new surfaces, and prefer Hydra configs or dataclasses for structured settings.

## Testing Guidelines
Pytest powers the suite; new tests belong under `tests/test_<feature>.py` with descriptive method names (e.g., `test_rppo_handles_bptt_masks`). Build dummy environments or fixtures, mirroring `tests/test_algorithms.py`, to keep stochastic behaviour deterministic. Ensure integration tests seed `jax.random` for reproducibility. For features needing accelerator hardware, guard them with `pytest.mark.skipif`.

## Commit & Pull Request Guidelines
Commit history favours short, imperative subjects (`Update deps`, `Move out benchmarks`). Keep the first line ≤72 characters and expand rationale in the body if necessary. Before opening a PR, confirm `uv run pytest` and `uv run pre-commit run --all-files` pass, link relevant issues, and summarise experiment results (include `outputs/<timestamp>/main.log` snippets or W&B links). Request reviews when you have clear reproduction steps and attach configuration diffs or command lines used.
