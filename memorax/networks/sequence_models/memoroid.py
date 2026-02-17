import math
from abc import abstractmethod
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.kernels.pallas import linear_recurrence as _raw_linear_recurrence
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel
from .utils import broadcast_mask, get_input_shape


def _get_nested_param(params, name):
    """Get parameter by '/'-separated path for nested module params."""
    for key in name.split("/"):
        params = params[key]
    return params


def _compute_phantom(sensitivity, param_indices, params):
    """Compute phantom gradient from sensitivities and parameter diffs.

    For params with an index array in param_indices (compressed jacobians),
    uses indexing to map param diffs to state dims.
    For params with extra dims in S (full jacobians), uses broadcasting.
    """
    if not sensitivity:
        return None
    first_S = next(iter(sensitivity.values()))
    phantom = jnp.zeros(first_S.shape[:3], dtype=first_S.dtype)
    for name, S in sensitivity.items():
        param = _get_nested_param(params, name)
        diff = param - jax.lax.stop_gradient(param)
        if name in param_indices:
            phantom = phantom + S * diff[param_indices[name]]
        else:
            phantom = phantom + jnp.sum(S * diff, axis=tuple(range(3, S.ndim)))
    return phantom


def _propagate_sensitivities(decay, jacobians, sensitivity, mask):
    """Propagate RTRL sensitivities: S_{t+1} = decay_t * S_t + J_t per parameter."""
    B, T, H = decay.shape
    next_sensitivity = {}

    for name in sorted(jacobians.keys()):
        J = jacobians[name]  # (B, T, H, *param_dims)
        S = sensitivity[name]  # (B, 1, H, *param_dims)
        param_size = math.prod(J.shape[3:])

        out_shape = J.shape[2:]
        J = J.reshape(B, T, H * param_size)
        S = S.reshape(B, 1, H * param_size)
        a = jnp.repeat(decay, param_size, axis=-1) if param_size > 1 else decay

        _, S = _raw_linear_recurrence(x=J, a=a, initial_carry=S, mask=mask)
        next_sensitivity[name] = S.reshape(B, 1, *out_shape)

    return next_sensitivity


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
        """Return (decay, jacobians) for RTRL sensitivity propagation.

        Args:
            carry: Full previous carry (state is always carry[0]),
                with each leaf having shape (B, T, *dims).
            z: Scan elements from __call__.
            inputs: Raw inputs.

        Returns:
            (decay, jacobians) where decay is (B, T, H_flat) diagonal
            state-to-state Jacobian and jacobians is a dict mapping
            parameter names to (B, T, H_flat, *param_dims) tensors.
        """
        return None

    def get_param_indices(self):
        """Return dict mapping parameter names to index arrays for compressed jacobians."""
        return {}

    def initialize_sensitivity(self, key, input_shape):
        """Return sensitivity_dict or None if RTRL unsupported."""
        return None


def associative_scan(cell, z, initial_carry, mask):
    z = jax.tree.map(
        lambda c, e: jnp.concatenate([c, e], axis=1),
        initial_carry,
        z,
    )

    reset = jnp.concatenate([jnp.zeros((mask.shape[0], 1)), mask], axis=1)
    reset = reset[..., None]

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

    next_carry = jax.tree.map(lambda s: s[:, -1:], h)
    h = jax.tree.map(lambda s: s[:, 1:], h)
    return h, next_carry


class Memoroid(SequenceModel):
    cell: MemoroidCellBase
    scan_fn: Callable = associative_scan

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
        h, next_carry = self.scan_fn(self.cell, z, initial_carry, mask)
        y = self.cell.read(h, inputs, **kwargs)

        return next_carry, y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)

    @nn.compact
    def local_jacobian(self, inputs, mask, carry, sensitivity=None, **kwargs):
        z = self.cell(inputs, **kwargs)
        param_indices = self.cell.get_param_indices()

        if sensitivity is not None:
            params = self.variables["params"]["cell"]
            phantom = _compute_phantom(sensitivity, param_indices, params)
            if phantom is not None:
                carry = (carry[0] + phantom.reshape(carry[0].shape), *carry[1:])

        h, next_carry = self.scan_fn(self.cell, z, carry, mask)
        y = self.cell.read(h, inputs, **kwargs)

        next_sensitivity = None
        if sensitivity is not None:
            prev_carry = jax.tree.map(
                lambda c, hh: jnp.concatenate([c, hh[:, :-1]], axis=1),
                carry,
                h,
            )
            decay, jacobians = self.cell.local_jacobian(prev_carry, z, inputs)
            if jacobians:
                next_sensitivity = _propagate_sensitivities(
                    decay, jacobians, sensitivity, mask
                )
            else:
                next_sensitivity = sensitivity

        return next_carry, y, next_sensitivity

    def initialize_sensitivity(self, key: jax.Array, input_shape: Tuple[int, ...]):
        return self.cell.initialize_sensitivity(key, input_shape)
