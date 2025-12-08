# Repository Guidelines

## Project Structure & Module Organization
Core source lives in `memorax/`, grouped by responsibility: `algorithms/` packages training loops for DQN, PPO, SAC, and memory-augmented variants; `networks/` houses recurrent cells and policy/value heads; `buffers/` and `utils/` provide rollout storage and shared helpers; `environments/` and `loggers/` wrap task definitions and logging sinks. The `popjaxrl/` tree mirrors experimental PopGym baselines with its own `algorithms/`, `envs/`, and `run_popgym.py`. Keep new tests under `tests/`, and direct generated checkpoints or Hydra logs to `outputs/`; long-form analysis lives in `plots/`.

## Build, Test, and Development Commands
- `uv sync` installs the core CPU stack specified in `pyproject.toml`; add `--extra cuda` to fetch GPU builds.
- `uv run main algorithm=dqn environment=cartpole` launches a reference experiment; override parameters Hydra-style (e.g., `logger=wandb seed=42`).
- `uv run python popjaxrl/run_popgym.py --num-runs 4 --env AliasPrevAction --arch s5` exercises the PopGym baselines.
- `uv run pytest` executes the full test suite; target individual files with `uv run pytest tests/test_algorithms.py`.

## Coding Style & Naming Conventions
Format Python with Black (line-length 88) and isort (`--profile black`); enforce linting through Flake8, Ruff, and MyPy (`--ignore-missing-imports`). Four-space indentation is standard, with type annotations expected on public APIs. Modules and functions use `snake_case`, classes use `PascalCase`, and constants remain `UPPER_CASE`. Run `pre-commit run --all-files` before committing to apply the configured checks and prettier formatting for Markdown.

## Testing Guidelines
Write pytest cases named `test_<feature>.py`, grouping scenario-specific helpers in fixtures. Favor deterministic seeds via JAX PRNG keys so regressions surface reliably. Exercise new environments or algorithms with smoke tests that verify forward passes and training loops complete a small rollout. Store any large artifacts under `outputs/` and exclude them from assertions. Use `uv run pytest -k "ppo"` when narrowing failures locally.

## Commit & Pull Request Guidelines
Repository history favors concise, imperative subjects (e.g., `Update models`, `Fix s/m/xLSTM`). Keep bodies short and reference issues when available. Before opening a pull request, confirm `uv run pytest` and `pre-commit run --all-files` pass, summarize the experiment command or config overrides you validated, and attach relevant WandB run links or plots. Highlight configuration or API changes in the PR description so downstream experiment scripts can be updated promptly.
