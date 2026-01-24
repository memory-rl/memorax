# Contributing to Memorax

Thank you for your interest in contributing to Memorax! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/memory-rl/memorax.git
   cd memorax
   ```

2. **Install dependencies with development tools**:
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**:
   ```bash
   uv run pytest
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality, all enforced via pre-commit hooks:

- **Black** - Code formatting
- **isort** - Import sorting
- **Flake8** - Linting (with bugbear plugin)
- **MyPy** - Type checking
- **Ruff** - Fast linting

Pre-commit hooks run automatically on `git commit`. To run them manually:

```bash
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_algorithms.py

# Run with verbose output
uv run pytest -v

# Run specific test class
uv run pytest tests/test_sequence_models.py::TestLRU
```

### Project Structure

```
memorax/
├── memorax/
│   ├── algorithms/      # RL algorithms (DQN, PPO, SAC, etc.)
│   ├── networks/        # Neural network components
│   │   ├── sequence_models/    # RNNs, SSMs, attention
│   │   ├── blocks/             # Building blocks (MoE, FFN, etc.)
│   │   ├── positional_embeddings/
│   │   └── heads/              # Action/value output heads
│   ├── environments/    # Environment wrappers
│   ├── buffers/         # Replay buffer utilities
│   ├── loggers/         # Logging backends
│   └── utils/           # Utility functions
├── examples/            # Runnable example scripts
└── tests/               # Test suite
```

## Contribution Guidelines

### Adding a New Sequence Model

1. Create a new file in `memorax/networks/sequence_models/`
2. Inherit from `nn.Module` and implement:
   - `initialize_carry(key, input_shape)` - Initialize hidden state
   - `__call__(inputs, mask, initial_carry)` - Forward pass
   - `combine(carry_a, carry_b)` - For parallel scan (if applicable)
3. Export in `memorax/networks/sequence_models/__init__.py`
4. Add tests in `tests/test_sequence_models.py`
5. Add an example in `examples/`

### Adding a New Algorithm

1. Create a new file in `memorax/algorithms/`
2. Implement:
   - Config dataclass with hyperparameters
   - State dataclass for training state
   - `init()`, `warmup()`, `train()`, `evaluate()` methods
3. Export in `memorax/algorithms/__init__.py`
4. Add tests in `tests/test_algorithms.py`
5. Add an example in `examples/`

### Adding Environment Support

1. Create a wrapper in `memorax/environments/`
2. Implement the `make(env_id)` function returning `(env, env_params)`
3. Register in `memorax/environments/environment.py`
4. Add tests in `tests/test_environments.py`

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run the test suite** to ensure nothing is broken:
   ```bash
   uv run pytest
   ```

5. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation if needed
- Ensure all pre-commit hooks pass
- Provide a clear description of changes

## Reporting Issues

When reporting issues, please include:

- Python version
- JAX version
- Operating system
- Minimal reproducible example
- Full error traceback

## Questions?

Feel free to open a GitHub issue for questions about contributing.

## License

By contributing to Memorax, you agree that your contributions will be licensed under the Apache License 2.0.
