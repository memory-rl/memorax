from abc import abstractmethod
from typing import Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.linen.recurrent import RNNCellBase as FlaxRNNCellBase
from flax.typing import InOutScanAxis

from memorax.utils.axes import (
    add_feature_axis,
    broadcast_mask,
    get_time_axis_and_input_shape,
    mask_carry,
)
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel


class RNNCellBase(FlaxRNNCellBase):
    @abstractmethod
    def local_jacobian(
        self, carry: Carry, inputs: Array, sensitivity: Dict[str, Array], **kwargs
    ) -> Tuple[Carry, Array, Dict[str, Array]]: ...

    def compute_phantom(self, sensitivity: Dict[str, Array]) -> Array:
        params = self.variables["params"]
        phantom = 0
        for name, S in sensitivity.items():
            param = params
            for key in name.split("/"):
                param = param[key]
            diff = param - jax.lax.stop_gradient(param)
            phantom = phantom + jnp.sum(S * diff, axis=tuple(range(3, S.ndim)))
        return phantom

    @abstractmethod
    def inject_phantom(self, carry: Carry, phantom: Array) -> Carry: ...

    @abstractmethod
    def initialize_sensitivity(
        self, key: jax.Array, input_shape: Tuple[int, ...]
    ) -> Optional[Dict[str, Array]]: ...


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

    def local_jacobian(self, inputs, mask, carry, sensitivity=None, **kwargs):
        if sensitivity is None:
            next_carry, y = self(inputs, mask, carry, **kwargs)
            return next_carry, y, None

        time_axis, input_shape = get_time_axis_and_input_shape(inputs)
        initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        def scan_fn(cell, state, x, mask_t):
            cell_carry, sensitivity = state

            phantom = cell.compute_phantom(sensitivity)
            cell_carry = cell.inject_phantom(cell_carry, phantom)

            cell_carry = mask_carry(mask_t, cell_carry, initial_carry)

            sensitivity = jax.tree.map(
                lambda s: jnp.where(broadcast_mask(add_feature_axis(mask_t), s), 0, s),
                sensitivity,
            )

            next_carry, y, next_sensitivity = cell.local_jacobian(
                cell_carry, x, sensitivity
            )

            return (next_carry, next_sensitivity), y

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

        (next_carry, next_sensitivity), outputs = scan(
            self.cell, (carry, sensitivity), inputs, mask
        )

        if self.features is not None:
            outputs = self.output_projection(outputs)

        return next_carry, outputs, next_sensitivity

    def initialize_sensitivity(self, key, input_shape):
        return self.cell.initialize_sensitivity(key, input_shape)
