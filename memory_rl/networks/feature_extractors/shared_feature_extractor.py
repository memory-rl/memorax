from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class SharedFeatureExtractor(nn.Module):

    extractor: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        reward: Optional[jnp.ndarray] = None,
        done: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        x = jnp.concatenate(
            [x for x in (observation, action, reward, done) if x is not None], axis=-1
        )
        return self.extractor(x)
