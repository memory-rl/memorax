"""Memorax: A unified JAX/Flax framework for memory-augmented reinforcement learning."""

__version__ = "1.0.0"

# Algorithms
from memorax.algorithms import (
    DQN,
    PPO,
    PQN,
    SAC,
    DQNConfig,
    DQNState,
    PPOConfig,
    PPOState,
    PQNConfig,
    PQNState,
    SACConfig,
    SACState,
)

# Environment factory
from memorax.environments import make

# Loggers
from memorax.loggers import (
    ConsoleLogger,
    DashboardLogger,
    FileLogger,
    Logger,
    LoggerState,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)

# Core network components
from memorax.networks import (
    CNN,
    MLP,
    FeatureExtractor,
    Network,
    SequenceModel,
    SequenceModelWrapper,
)

__all__ = [
    # Version
    "__version__",
    # Algorithms
    "DQN",
    "DQNConfig",
    "DQNState",
    "PPO",
    "PPOConfig",
    "PPOState",
    "PQN",
    "PQNConfig",
    "PQNState",
    "SAC",
    "SACConfig",
    "SACState",
    # Environment
    "make",
    # Networks
    "CNN",
    "MLP",
    "Network",
    "FeatureExtractor",
    "SequenceModel",
    "SequenceModelWrapper",
    # Loggers
    "Logger",
    "LoggerState",
    "ConsoleLogger",
    "DashboardLogger",
    "FileLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "NeptuneLogger",
]
