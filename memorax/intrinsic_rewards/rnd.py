from typing import Any

import jax
import jax.numpy as jnp
from flax import core
from flax import linen as nn

from memorax.utils.typing import Array


class RND(nn.Module):
    """Random Network Distillation.

    Implements the intrinsic reward interface:
    - intrinsic_reward(params, obs, next_obs, action) -> rewards
    - loss(params, obs, next_obs, action) -> (loss, metrics)

    Note: RND only uses obs, not next_obs or action. The interface
    is kept consistent with ICM for compatibility with IRPPO.

    Args:
        target: Network for computing target features (frozen after init).
        predictor: Network for predicting target features (trained).
    """

    target: nn.Module
    predictor: nn.Module

    @nn.compact
    def __call__(self, obs: Array, next_obs: Array, action: Array):
        """Forward pass for initialization."""
        target_features = self.target(obs)
        predicted_features = self.predictor(obs)
        return target_features, predicted_features

    def intrinsic_reward(
        self,
        params: core.FrozenDict[str, Any],
        obs: Array,
        next_obs: Array,
        action: Array,
    ) -> Array:
        """Compute intrinsic reward as prediction error."""
        target_features = self.target.apply({"params": params["target"]}, obs)
        predicted_features = self.predictor.apply({"params": params["predictor"]}, obs)

        intrinsic_reward = jnp.square(target_features - predicted_features).mean(
            axis=-1
        )
        return intrinsic_reward

    def loss(
        self,
        params: core.FrozenDict[str, Any],
        obs: Array,
        next_obs: Array,
        action: Array,
    ) -> tuple[Array, dict]:
        """Compute RND loss (predictor tries to match frozen target)."""
        target_features = jax.lax.stop_gradient(
            self.target.apply({"params": params["target"]}, obs)
        )
        predicted_features = self.predictor.apply({"params": params["predictor"]}, obs)

        loss = jnp.square(target_features - predicted_features).mean()

        return loss, {"losses/ir_rnd_loss": loss}
