from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class FeatureExtractor(nn.Module):

    observation_extractor: nn.Module
    action_extractor: Optional[nn.Module] = None
    reward_extractor: Optional[nn.Module] = None
    concatenate: bool = True

    def extract(
        self,
        features: tuple,
        extractor: Optional[nn.Module],
        x: Optional[jnp.ndarray] = None,
    ):
        if extractor is not None and x is not None:
            features += (extractor(x),)

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        reward: Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        features = (self.observation_extractor(observation),)
        self.extract(features, self.action_extractor, action)
        self.extract(features, self.reward_extractor, reward)

        if self.concatenate:
            features = jnp.concatenate(features, axis=-1)

        return features
