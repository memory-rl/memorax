from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class FeatureExtractor(nn.Module):
    observation_extractor: nn.Module
    action_extractor: Optional[nn.Module] = None
    reward_extractor: Optional[nn.Module] = None
    done_extractor: Optional[nn.Module] = None

    def extract(
        self,
        features: list,
        extractor: Optional[nn.Module],
        x: Optional[jnp.ndarray] = None,
    ):
        if extractor is not None and x is not None:
            features.append(extractor(x))

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        **kwargs,
    ):
        features = [self.observation_extractor(observation)]
        self.extract(features, self.action_extractor, action)
        self.extract(features, self.reward_extractor, reward)
        self.extract(features, self.done_extractor, done.astype(jnp.int32))

        features = jnp.concatenate(features, axis=-1)

        return features
