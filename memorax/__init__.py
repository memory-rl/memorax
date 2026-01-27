"""Memorax: A unified JAX/Flax framework for memory-augmented reinforcement learning."""

__version__ = "1.0.1"

# Algorithms
from memorax.algorithms import (
    DQN,
    IPPO,
    IRPPO,
    PPO,
    PQN,
    R2D2,
    SAC,
    DQNConfig,
    DQNState,
    IPPOConfig,
    IPPOState,
    IRPPOConfig,
    IRPPOState,
    PPOConfig,
    PPOState,
    PQNConfig,
    PQNState,
    R2D2Config,
    R2D2State,
    SACConfig,
    SACState,
)

# Intrinsic Rewards
from memorax.intrinsic_rewards import ICM, RND

# Environment factory
from memorax.environments import make

# Intrinsic Rewards
from memorax.intrinsic_rewards import ICM, RND

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
    "IPPO",
    "IPPOConfig",
    "IPPOState",
    "IRPPO",
    "IRPPOConfig",
    "IRPPOState",
    "PPO",
    "PPOConfig",
    "PPOState",
    "PQN",
    "PQNConfig",
    "PQNState",
    "R2D2",
    "R2D2Config",
    "R2D2State",
    "SAC",
    "SACConfig",
    "SACState",
    # Intrinsic Rewards
    "ICM",
    "RND",
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
