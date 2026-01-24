from typing import Callable, Optional

import flax.linen as nn

from memorax.utils.typing import Array, Carry

from .base import Block


class PreNorm(nn.Module, Block):
    """Applies normalization before the module: output = module(norm(x)).

    Args:
        module: The module to wrap.
        norm: Normalization class (default: nn.LayerNorm).
        norm_kwargs: Additional kwargs passed to the norm constructor.
    """

    module: nn.Module
    norm: Callable = nn.LayerNorm

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        x = self.norm()(inputs)
        return self.module(x, mask=mask, initial_carry=initial_carry, **kwargs)

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.module.initialize_carry(key, input_shape)


class PostNorm(nn.Module, Block):
    """Applies normalization after the module: output = norm(module(x)).

    Args:
        module: The module to wrap.
        norm: Normalization class (default: nn.LayerNorm).
        norm_kwargs: Additional kwargs passed to the norm constructor.
    """

    module: nn.Module
    norm: Callable = nn.LayerNorm

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        carry, output = self.module(
            inputs, mask=mask, initial_carry=initial_carry, **kwargs
        )
        return carry, self.norm()(output)

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.module.initialize_carry(key, input_shape)
