# Installation

## Basic Installation

Install Memorax using pip:

```bash
pip install memorax
```

Or using uv:

```bash
uv add memorax
```

## Optional Dependencies

Memorax supports various environment backends. Install the ones you need:

```bash
# Brax physics environments
pip install memorax[brax]

# Craftax multi-task environments
pip install memorax[craftax]

# Navigation environments
pip install memorax[navix]

# POPGym Arcade
pip install memorax[popgym-arcade]

# All environments
pip install memorax[all]
```

## GPU Support

For GPU acceleration with CUDA:

```bash
pip install memorax[cuda]
```

## Development Installation

To contribute to Memorax:

```bash
git clone https://github.com/noahfarr/memorax.git
cd memorax
uv sync
uv run pre-commit install
```

## Verifying Installation

```python
import memorax
from memorax.algorithms import PPO
from memorax.environments import make

# Create a test environment
env, env_params = make("gymnax::CartPole-v1")
print(f"Memorax version: {memorax.__version__}")
print(f"Environment created: {env}")
```
