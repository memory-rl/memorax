from typing import Mapping

import flax.linen as nn
from flax.linen.recurrent import Carry
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.typing import InOutScanAxis, Initializer
import jax

from memorax.networks.recurrent.utils import (
    get_time_axis_and_input_shape,
    mask_carry,
)


class RNN(nn.Module):

    cell: nn.RNNCellBase
    unroll: int = 1
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = FrozenDict()
    variable_broadcast: CollectionFilter = "params"
    variable_carry: CollectionFilter = False
    split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict({"params": False})
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: Carry,
    ):
        time_axis, input_shape = get_time_axis_and_input_shape(inputs)

        carry: Carry = initial_carry

        # initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        def scan_fn(cell, carry, x, mask):
            carry = mask_carry(
                mask, carry, self.cell.initialize_carry(jax.random.key(0), input_shape)
            )
            carry, y = cell(carry, x)
            return carry, y

        scan = nn.transforms.scan(
            scan_fn,
            in_axes=time_axis,
            out_axes=time_axis,
            unroll=self.unroll,
            variable_axes=self.variable_axes,
            variable_broadcast=self.variable_broadcast,
            variable_carry=self.variable_carry,
            split_rngs=self.split_rngs,
        )

        carry, outputs = scan(self.cell, carry, inputs, mask)

        return carry, outputs

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.cell.initialize_carry(key, input_shape)
