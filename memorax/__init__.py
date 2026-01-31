"""Memorax: A unified JAX/Flax framework for memory-augmented reinforcement learning."""

__version__ = "1.0.1"

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

from memorax.intrinsic_rewards import ICM, RND

from memorax.environments import make

from memorax.intrinsic_rewards import ICM, RND


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

from memorax.networks import (
    CNN,
    MLP,
    FeatureExtractor,
    Network,
    SequenceModel,
    SequenceModelWrapper,
)

__all__ = [
    "__version__",
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
    "ICM",
    "RND",
    "make",
    "CNN",
    "MLP",
    "Network",
    "FeatureExtractor",
    "SequenceModel",
    "SequenceModelWrapper",
    "Logger",
    "LoggerState",
    "ConsoleLogger",
    "DashboardLogger",
    "FileLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "NeptuneLogger",
]
