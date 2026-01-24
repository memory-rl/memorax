"""Memorax: A unified JAX/Flax framework for memory-augmented reinforcement learning."""

__version__ = "0.1.0"

# Algorithms
from memorax.algorithms import (
    DQN,
    DQNConfig,
    DQNState,
    PPO,
    PPOConfig,
    PPOState,
    PQN,
    PQNConfig,
    PQNState,
    SAC,
    SACConfig,
    SACState,
)

# Environment factory
from memorax.environments import make

# Core network components
from memorax.networks import (
    CNN,
    MLP,
    Network,
    FeatureExtractor,
    SequenceModel,
    SequenceModelWrapper,
)

# Loggers
from memorax.loggers import (
    Logger,
    LoggerState,
    ConsoleLogger,
    DashboardLogger,
    FileLogger,
    TensorBoardLogger,
    WandbLogger,
    NeptuneLogger,
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
