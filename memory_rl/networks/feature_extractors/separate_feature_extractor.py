from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class SeparateFeatureExtractor(nn.Module):

    observation_extractor: nn.Module
    action_extractor: Optional[nn.Module] = None
    reward_extractor: Optional[nn.Module] = None
    done_extractor: Optional[nn.Module] = None

    def extract(
        self,
        feats: list,
        extractor: Optional[nn.Module],
        x: Optional[jnp.ndarray] = None,
    ):
        if extractor is not None and x is not None:
            feats.append(extractor(x))

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        reward: Optional[jnp.ndarray] = None,
        done: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        feats = [self.observation_extractor(observation)]
        self.extract(feats, self.action_extractor, action)
        self.extract(feats, self.reward_extractor, reward)
        self.extract(feats, self.done_extractor, done)
        return jnp.concatenate(feats, axis=-1)
