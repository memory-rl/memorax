import math
from abc import abstractmethod
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.axes import (
    add_feature_axis,
    broadcast_mask,
    get_input_shape,
    init,
    last,
    tail,
)
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel


class MemoroidCellBase(nn.Module):
    @abstractmethod
    def __call__(self, x: Array, **kwargs) -> Carry: ...

    @abstractmethod
    def binary_operator(self, a: Carry, b: Carry) -> Carry: ...

    @abstractmethod
    def read(self, h: Carry, x: Array, **kwargs) -> Array: ...

    @abstractmethod
    def initialize_carry(
        self, key: jax.Array, input_shape: Tuple[int, ...]
    ) -> Carry: ...

    def local_jacobian(self, carry, z, inputs, **kwargs):
        return None

    def get_param_indices(self):
        return {}

    def initialize_sensitivity(self, key, input_shape):
        return None


class Memoroid(SequenceModel):
    cell: MemoroidCellBase

    def scan_fn(self, z, initial_carry, mask):
        z = jax.tree.map(
            lambda c, e: jnp.concatenate([c, e], axis=1),
            initial_carry,
            z,
        )

        reset = jnp.concatenate([jnp.zeros((mask.shape[0], 1)), mask], axis=1)
        reset = add_feature_axis(reset)

        cell = self.cell

        @jax.vmap
        def binary_operator(lhs, rhs):
            lhs_carry, lhs_reset = lhs
            rhs_carry, rhs_reset = rhs

            combined = cell.binary_operator(lhs_carry, rhs_carry)

            out = jax.tree.map(
                lambda rc, c: jnp.where(broadcast_mask(rhs_reset, rc), rc, c),
                rhs_carry,
                combined,
            )

            return out, jnp.maximum(lhs_reset, rhs_reset)

        h, _ = jax.lax.associative_scan(binary_operator, (z, reset), axis=1)

        next_carry = jax.tree.map(last, h)
        h = jax.tree.map(tail, h)
        return h, next_carry

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> Tuple[Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        z = self.cell(inputs, **kwargs)
        h, next_carry = self.scan_fn(z, initial_carry, mask)
        y = self.cell.read(h, inputs, **kwargs)

        return next_carry, y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)

    def _compute_phantom(self, sensitivity, param_indices, params):
        if not sensitivity:
            return None
        phantom = 0
        for name, S in sensitivity.items():
            param = params
            for key in name.split("/"):
                param = param[key]
            diff = param - jax.lax.stop_gradient(param)
            if name in param_indices:
                phantom = phantom + S * diff[param_indices[name]]
            else:
                phantom = phantom + jnp.sum(S * diff, axis=tuple(range(3, S.ndim)))
        return phantom

    def _propagate_sensitivities(self, decay, jacobians, sensitivity, mask):
        B, T, H = decay.shape
        mask = add_feature_axis(mask)
        next_sensitivity = {}

        @jax.vmap
        def binary_operator(a, b):
            state_i, decay_i = a
            state_j, decay_j = b
            return (decay_j * state_i + state_j, decay_j * decay_i)

        for name in sorted(jacobians.keys()):
            J = jacobians[name]
            S = sensitivity[name]
            _, _, _, *param_shape = J.shape
            param_size = math.prod(param_shape)

            J = J.reshape(B, T, H * param_size)
            S = S.reshape(B, 1, H * param_size)
            a = jnp.where(mask, 0, jnp.repeat(decay, param_size, axis=-1))

            state = jnp.concatenate([S, J], axis=1)
            a = jnp.concatenate([jnp.ones_like(S), a], axis=1)

            state, _ = jax.lax.associative_scan(
                binary_operator, (state, a), axis=1
            )
            next_sensitivity[name] = last(state).reshape(B, 1, H, *param_shape)

        return next_sensitivity

    @nn.compact
    def local_jacobian(self, inputs, mask, carry, sensitivity=None, **kwargs):
        z = self.cell(inputs, **kwargs)
        param_indices = self.cell.get_param_indices()

        if sensitivity is not None:
            phantom = self._compute_phantom(
                sensitivity, param_indices, self.variables["params"]["cell"]
            )
            if phantom is not None:
                state, *rest = carry
                carry = (state + phantom, *rest)

        h, next_carry = self.scan_fn(z, carry, mask)
        y = self.cell.read(h, inputs, **kwargs)

        next_sensitivity = None
        if sensitivity is not None:
            prev_carry = jax.tree.map(
                lambda initial_carry, hidden_states: jnp.concatenate(
                    [initial_carry, init(hidden_states)], axis=1
                ),
                carry,
                h,
            )
            decay, jacobians = self.cell.local_jacobian(prev_carry, z, inputs)
            if jacobians:
                next_sensitivity = self._propagate_sensitivities(
                    decay, jacobians, sensitivity, mask
                )
            else:
                next_sensitivity = sensitivity

        return next_carry, y, next_sensitivity

    def initialize_sensitivity(self, key: jax.Array, input_shape: Tuple[int, ...]):
        return self.cell.initialize_sensitivity(key, input_shape)
