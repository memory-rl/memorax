from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import core
from flax import linen as nn

from memorax.utils.typing import Array


class ICM(nn.Module):
    """Intrinsic Curiosity Module.

    Implements the intrinsic reward interface:
    - intrinsic_reward(params, obs, next_obs, action) -> rewards
    - loss(params, obs, next_obs, action) -> (loss, metrics)

    Args:
        encoder: Network for encoding observations to features.
        forward_model: Network for predicting next features (input: concat of features and one-hot action).
        inverse_model: Network for predicting action logits (input: concat of features and next_features).
        num_actions: Number of actions for one-hot encoding.
        beta: Weight for forward loss vs inverse loss (default 0.2).
    """

    encoder: nn.Module
    forward_model: nn.Module
    inverse_model: nn.Module
    num_actions: int
    beta: float = 0.2

    def encode_action(self, action: Array) -> Array:
        if action.dtype in (jnp.int32, jnp.int64, jnp.uint32, jnp.uint64):
            return jax.nn.one_hot(action, self.num_actions)
        return action

    @nn.compact
    def __call__(self, obs: Array, next_obs: Array, action: Array):
        """Forward pass for initialization."""
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)
        forward_input = jnp.concatenate([features, self.encode_action(action)], axis=-1)
        inverse_input = jnp.concatenate([features, next_features], axis=-1)
        predicted_next_features = self.forward_model(forward_input)
        predicted_action_logits = self.inverse_model(inverse_input)
        return features, next_features, predicted_next_features, predicted_action_logits

    def intrinsic_reward(
        self,
        params: core.FrozenDict[str, Any],
        obs: Array,
        next_obs: Array,
        action: Array,
    ) -> Array:
        """Compute intrinsic reward as forward model prediction error."""
        features = self.encoder.apply({"params": params["encoder"]}, obs)
        next_features = self.encoder.apply({"params": params["encoder"]}, next_obs)
        forward_input = jnp.concatenate([features, self.encode_action(action)], axis=-1)
        predicted_next_features = self.forward_model.apply(
            {"params": params["forward_model"]}, forward_input
        )
        intrinsic_reward = jnp.square(next_features - predicted_next_features).mean(
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
        """Compute ICM loss (forward + inverse)."""
        features = self.encoder.apply({"params": params["encoder"]}, obs)
        next_features = self.encoder.apply({"params": params["encoder"]}, next_obs)

        forward_input = jnp.concatenate([features, self.encode_action(action)], axis=-1)
        predicted_next_features = self.forward_model.apply(
            {"params": params["forward_model"]}, forward_input
        )
        forward_loss = jnp.square(
            jax.lax.stop_gradient(next_features) - predicted_next_features
        ).mean()

        inverse_input = jnp.concatenate([features, next_features], axis=-1)
        predicted_action_logits = self.inverse_model.apply(
            {"params": params["inverse_model"]}, inverse_input
        )
        inverse_loss = optax.softmax_cross_entropy_with_integer_labels(
            predicted_action_logits, action
        ).mean()

        total_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss

        return total_loss, {
            "losses/ir_forward_loss": forward_loss,
            "losses/ir_inverse_loss": inverse_loss,
            "losses/ir_total_loss": total_loss,
        }
