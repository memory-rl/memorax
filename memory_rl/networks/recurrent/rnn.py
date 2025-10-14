import flax.linen as nn
from flax.linen.recurrent import Carry
import jax

from memory_rl.networks.recurrent.utils import (
    get_time_axis_and_input_shape,
    mask_carry,
)


class RNN(nn.Module):

    cell: nn.RNNCellBase
    unroll: int = 1

    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        *,
        initial_carry: Carry | None = None,
        **kwargs,
    ):
        time_axis, input_shape = get_time_axis_and_input_shape(inputs)

        carry: Carry
        if initial_carry is None:
            init_key = jax.random.key(0)

            carry = self.cell.initialize_carry(init_key, input_shape)
        else:
            carry = initial_carry

        initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        def scan_fn(carry, x):
            carry = mask_carry(mask, carry, initial_carry)
            carry, y = self.cell(carry, x)
            return carry, y

        scan = nn.transforms.scan(
            scan_fn,
            in_axes=time_axis,
            out_axes=time_axis,
            unroll=self.unroll,
        )

        carry, outputs = scan(carry, inputs)

        return carry, outputs
