from typing import Mapping, Optional, Tuple

import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.typing import InOutScanAxis

from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel
from .utils import get_time_axis_and_input_shape, mask_carry


class RNN(SequenceModel):
    cell: nn.RNNCellBase
    features: Optional[int] = None
    unroll: int = 1
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = FrozenDict()
    variable_broadcast: CollectionFilter = "params"
    variable_carry: CollectionFilter = False
    split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict({"params": False})

    def setup(self):
        if self.features is not None:
            self.output_projection = nn.Dense(self.features)

    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> Tuple[Carry, Array]:
        time_axis, input_shape = get_time_axis_and_input_shape(inputs)

        if initial_carry is None:
            initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        carry: Carry = initial_carry

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

        if self.features is not None:
            outputs = self.output_projection(outputs)

        return carry, outputs

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
