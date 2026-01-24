from typing import Callable, Optional

import flax.linen as nn

from memorax.utils.typing import Array, Carry


class Residual(nn.Module):
    """Wraps a module with a residual connection: output = x + module(x)."""

    module: nn.Module

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
        return carry, inputs + output


class GatedResidual(nn.Module):
    """Residual connection with a learned gate: output = x + gate * module(x)."""

    module: nn.Module
    gate: Callable = nn.sigmoid

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        features = inputs.shape[-1]

        carry, output = self.module(
            inputs, mask=mask, initial_carry=initial_carry, **kwargs
        )

        gate = nn.Dense(
            features,
            name="gate",
        )(inputs)
        gate = self.gate(gate)

        return carry, inputs + gate * output
