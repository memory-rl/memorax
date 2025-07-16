from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig


class Network(nn.Module):
    head: nn.Module
    feature_extractor: Optional[nn.Module] = None
    torso: Optional[nn.Module] = None

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        initial_carry: Optional[jnp.ndarray] = None,
        return_carry_history: bool = False,
    ):
        x = (
            jnp.concatenate([observation, action], axis=-1)
            if action is not None
            else observation
        )

        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        if self.torso is not None:
            x = self.torso(
                x,
                mask=mask,
                initial_carry=initial_carry,
                return_carry_history=return_carry_history,
            )

        if isinstance(x, tuple):
            h, x = x
            return h, self.head(x)
        else:
            return self.head(x)
