from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig



class Network(nn.Module):
    feature_extractor: nn.Module
    head: nn.Module
    torso: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observation: jnp.ndarray, *_, action: Optional[jnp.ndarray] = None, **kwargs):
        x = jnp.concatenate([observation, action], axis=-1) if action is not None else observation
        x = self.feature_extractor(x)
        if self.torso is not None:
            x = self.torso(x, **kwargs)

            if isinstance(x, tuple):
                h, x = x
                return h, self.head(x, **kwargs)
                
        return self.head(x, **kwargs)

