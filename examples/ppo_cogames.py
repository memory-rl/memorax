import time
from dataclasses import asdict
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pufferlib
from cogames.cli.mission import get_mission
from cogames.cogs_vs_clips.cogsguard_reward_variants import apply_reward_variants
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator

from memorax.algorithms import PPO, PPOConfig
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import CNN, MLP, FeatureExtractor, Network, heads
from memorax.utils.wrappers import PufferLibWrapper


# =============================================================================
# Observation Shim: Token to Box Conversion
# =============================================================================
# MettaGrid observations are sparse token-based tensors of shape (num_tokens, 3)
# where each token is (coord, attr_idx, attr_val) as uint8:
#   - coord: encodes (row, col) position in a single byte (high nibble = row, low nibble = col)
#   - attr_idx: feature/attribute index
#   - attr_val: the value for that attribute
#
# This converts to dense box format (height, width, channels) suitable for CNNs.
# =============================================================================


class ObsTokenToBox(nn.Module):
    """Converts token observations to dense box format for CNNs.

    Args:
        num_layers: Number of feature channels in the output (max attr_idx + 1)
        obs_width: Width of the egocentric observation grid
        obs_height: Height of the egocentric observation grid
        normalize: Whether to normalize output by feature normalizations
        feature_normalizations: Optional tuple of (attr_idx, normalization) pairs
    """

    num_layers: int
    obs_width: int
    obs_height: int
    normalize: bool = True
    feature_normalizations: Optional[tuple[tuple[int, float], ...]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Convert token observations to box format.

        Args:
            x: Token observations of shape (..., num_tokens, 3)
               Each token is (coord_byte, attr_idx, attr_val) as uint8

        Returns:
            Box observations of shape (..., obs_height, obs_width, num_layers)
            Ready for CNN processing (NHWC format for JAX convolutions)
        """
        # Store original shape for later reshaping
        *batch_dims, num_tokens, token_dim = x.shape
        batch_size = 1
        for d in batch_dims:
            batch_size *= d

        # Flatten batch dimensions
        x = x.reshape(batch_size, num_tokens, token_dim)

        # Extract token components
        coords_byte = x[..., 0].astype(jnp.uint8)  # (B, M)
        attr_indices = x[..., 1].astype(jnp.int32)  # (B, M)
        attr_values = x[..., 2].astype(jnp.float32)  # (B, M)

        # Decode coordinates from packed byte (low nibble = x/col, high nibble = y/row)
        x_coords = (coords_byte & 0x0F).astype(jnp.int32)  # Low nibble -> x/col
        y_coords = (coords_byte >> 4).astype(jnp.int32)  # High nibble -> y/row

        # Create mask for valid tokens (coord_byte != 0xFF indicates valid token)
        valid_mask = coords_byte != 0xFF

        # Apply feature normalizations if provided
        if self.normalize and self.feature_normalizations is not None:
            # Build normalization tensor
            norm_factors = jnp.ones(256, dtype=jnp.float32)
            for idx, norm in self.feature_normalizations:
                norm_factors = norm_factors.at[idx].set(norm)

            # Look up normalization factors for each token's attr_idx
            token_norms = norm_factors[attr_indices]
            attr_values = attr_values / token_norms

        # Compute linear indices for scatter operation
        # Index = attr_idx * (width * height) + x_coord * height + y_coord
        dim_per_layer = self.obs_width * self.obs_height
        flat_spatial_index = x_coords * self.obs_height + y_coords
        combined_index = attr_indices * dim_per_layer + flat_spatial_index

        # Mask out invalid entries
        safe_index = jnp.where(valid_mask, combined_index, jnp.zeros_like(combined_index))
        safe_values = jnp.where(valid_mask, attr_values, jnp.zeros_like(attr_values))

        # Clamp indices to valid range to avoid out-of-bounds
        max_index = self.num_layers * dim_per_layer - 1
        safe_index = jnp.clip(safe_index, 0, max_index)

        def scatter_single(indices, values):
            """Scatter values to indices in a single flattened buffer."""
            total_size = self.num_layers * dim_per_layer
            return jnp.zeros(total_size, dtype=jnp.float32).at[indices].add(values)

        # Apply scatter for each batch element
        box_flat = jax.vmap(scatter_single)(safe_index, safe_values)

        # Reshape to (B, num_layers, width, height)
        box_obs = box_flat.reshape(batch_size, self.num_layers, self.obs_width, self.obs_height)

        # Transpose to NHWC format for JAX convolutions: (B, H, W, C)
        box_obs = jnp.transpose(box_obs, (0, 3, 2, 1))

        # Restore original batch dimensions
        output_shape = (*batch_dims, self.obs_height, self.obs_width, self.num_layers)
        box_obs = box_obs.reshape(output_shape)

        return box_obs


class ObsShimCNN(nn.Module):
    """Combines token-to-box conversion with a CNN feature extractor.

    Args:
        num_layers: Number of feature channels (from obs_features)
        obs_width: Width of the egocentric observation grid
        obs_height: Height of the egocentric observation grid
        cnn: CNN module to apply after conversion
        normalize_tokens: Whether to normalize token values by feature normalizations
        feature_normalizations: Tuple of (attr_idx, normalization) pairs
    """

    num_layers: int
    obs_width: int
    obs_height: int
    cnn: nn.Module
    normalize_tokens: bool = True
    feature_normalizations: Optional[tuple[tuple[int, float], ...]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Process token observations through token-to-box shim and CNN.

        Args:
            x: Token observations of shape (batch, seq, num_tokens, 3)

        Returns:
            CNN features of shape (batch, seq, feature_dim)
        """
        batch_size, sequence_length, *_ = x.shape

        # Convert tokens to box format: (batch, seq, num_tokens, 3) -> (batch, seq, H, W, C)
        box_obs = ObsTokenToBox(
            num_layers=self.num_layers,
            obs_width=self.obs_width,
            obs_height=self.obs_height,
            normalize=self.normalize_tokens,
            feature_normalizations=self.feature_normalizations,
        )(x)

        # Flatten batch and sequence for CNN: (batch, seq, H, W, C) -> (batch*seq, H, W, C)
        box_obs = box_obs.reshape(batch_size * sequence_length, *box_obs.shape[2:])

        # Apply CNN (expects batch, height, width, channels)
        features = self.cnn(box_obs, **kwargs)

        # Reshape back: (batch*seq, feature_dim) -> (batch, seq, feature_dim)
        features = features.reshape(batch_size, sequence_length, -1)

        return features


# =============================================================================
# End Observation Shim
# =============================================================================

total_timesteps = 50_000_000
min_steps_per_env = 10_000  # Minimum steps per env per training call

seed = 0
num_envs = 8  # Number of parallel environments


def make(env_id, variants=None, **kwargs):
    _, cfg, _ = get_mission(env_id)

    if variants:
        apply_reward_variants(cfg, variants=variants)

    simulator = Simulator()
    return MettaGridPufferEnv(simulator, cfg, **kwargs)


_, env_cfg, _ = get_mission("cogsguard_arena.basic")

# Get policy environment interface for observation parameters
policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

# Extract observation parameters for the shim
obs_features = list(policy_env_info.obs_features)
num_obs_layers = max((feat.id for feat in obs_features), default=-1) + 1
# Use tuple of tuples for hashability (required for JAX JIT with Flax modules)
feature_normalizations = tuple((feat.id, feat.normalization) for feat in obs_features)
obs_width = policy_env_info.obs_width
obs_height = policy_env_info.obs_height

print(f"Observation parameters: layers={num_obs_layers}, width={obs_width}, height={obs_height}")

simulator = Simulator()
puffer_env = pufferlib.vector.make(
    partial(make, variants=["credit"]),
    num_envs=num_envs,
    backend=pufferlib.vector.Serial,
    env_kwargs={"env_id": "cogsguard_arena.basic"},
)

env = PufferLibWrapper(puffer_env)
env_params = env.default_params


cfg = PPOConfig(
    name="PPO",
    num_envs=env.num_envs,  # Use num_envs from the env as this is num_envs x num_agents
    num_eval_envs=0,
    num_steps=64,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

num_train_steps = 10_000 * env.num_envs

# CNN for processing box observations
# Input will be (batch, seq, height, width, num_obs_layers) after shim conversion
cnn = CNN(
    features=(32, 64, 64),
    kernel_sizes=((3, 3), (3, 3), (3, 3)),
    strides=(1, 1, 1),
    padding="SAME",
    normalize=False,  # We normalize in the shim
)

# Observation extractor: token-to-box shim + CNN
observation_extractor = ObsShimCNN(
    num_layers=num_obs_layers,
    obs_width=obs_width,
    obs_height=obs_height,
    cnn=cnn,
    normalize_tokens=True,
    feature_normalizations=feature_normalizations,
)

feature_extractor = FeatureExtractor(
    observation_extractor=observation_extractor,
)
torso = MLP(features=(128,))
actor_network = nn.remat(Network)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.num_actions,
    ),
)

critic_network = nn.remat(Network)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)


key = jax.random.key(seed)

agent = PPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="PPO PufferLib CartPole", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg=asdict(cfg))

key, state = agent.init(key)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    key, state, transitions = agent.train(key, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = Logger.get_episode_statistics(transitions, "training")
    training_statistics = jax.tree.map(lambda x: x[None], training_statistics)
    info = jax.tree.map(lambda x: x.mean(), transitions.info)
    info = jax.tree.map(lambda x: x[None], info)
    data = {"training/SPS": SPS, **training_statistics, **info}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
