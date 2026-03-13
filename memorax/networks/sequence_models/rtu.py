from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal

from memorax.utils.typing import Array, Carry

from .rnn import RNNCellBase


def _initialize_nu_log(key, shape, r_min=0.0, r_max=1.0):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _initialize_theta_log(key, shape, max_phase=6.28):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(max_phase * u)


class RTUCell(RNNCellBase):

    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    eps: float = 1e-8

    @property
    def num_feature_axes(self):
        return 1

    def setup(self):
        self.nu_log = self.param(
            "nu_log",
            partial(_initialize_nu_log, r_min=self.r_min, r_max=self.r_max),
            (self.hidden_dim,),
        )
        self.theta_log = self.param(
            "theta_log",
            partial(_initialize_theta_log, max_phase=self.max_phase),
            (self.hidden_dim,),
        )
        self.B_real = self.param(
            "B_real",
            normal(stddev=1.0 / jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )
        self.B_imag = self.param(
            "B_imag",
            normal(stddev=1.0 / jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )

    def _g_phi_norm(self):
        r = jnp.exp(-jnp.exp(self.nu_log))
        theta = jnp.exp(self.theta_log)
        g = r * jnp.cos(theta)
        phi = r * jnp.sin(theta)
        norm = jnp.sqrt(1 - r**2) + self.eps
        return g, phi, norm, r

    @nn.compact
    def __call__(self, carry: Carry, inputs: Array) -> Tuple[Carry, Array]:
        H = self.hidden_dim
        h_c1, h_c2 = carry[..., :H], carry[..., H:]
        g, phi, norm, _ = self._g_phi_norm()

        pre1 = g * h_c1 - phi * h_c2 + norm * (inputs @ self.B_real.T)
        pre2 = g * h_c2 + phi * h_c1 + norm * (inputs @ self.B_imag.T)
        new_carry = jnp.concatenate([nn.relu(pre1), nn.relu(pre2)], axis=-1)
        return new_carry, new_carry

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        return jnp.zeros((*batch_dims, 2 * self.hidden_dim))

    def inject_phantom(self, carry: Carry, phantom: Array) -> Carry:
        return jax.lax.stop_gradient(carry) + phantom.reshape(carry.shape)

    def local_jacobian(
        self,
        carry: Carry,
        inputs: Array,
        sensitivity: Dict[str, Array],
        **kwargs,
    ) -> Tuple[Carry, Array, Dict[str, Array]]:
        H = self.hidden_dim
        h_c1, h_c2 = carry[..., :H], carry[..., H:]
        g, phi, norm, r = self._g_phi_norm()

        u_c1 = inputs @ self.B_real.T
        u_c2 = inputs @ self.B_imag.T
        pre1 = g * h_c1 - phi * h_c2 + norm * u_c1
        pre2 = g * h_c2 + phi * h_c1 + norm * u_c2
        d1 = (pre1 > 0).astype(carry.dtype)
        d2 = (pre2 > 0).astype(carry.dtype)
        new_carry = jnp.concatenate([nn.relu(pre1), nn.relu(pre2)], axis=-1)

        A = jnp.stack([jnp.stack([g, -phi]), jnp.stack([phi, g])])  # (2, 2, H)
        d = jnp.stack([d1, d2], axis=1)  # (B, 2, H)

        exp_nu = jnp.exp(self.nu_log)
        dg_dnu = -exp_nu * g
        dphi_dnu = -exp_nu * phi
        dnorm_dnu = exp_nu * r**2 / (jnp.sqrt(1 - r**2) + 1e-12)

        theta = jnp.exp(self.theta_log)
        dg_dtheta = -phi * theta
        dphi_dtheta = g * theta

        Bu = jnp.einsum('h,bf->bhf', norm, inputs)
        zeros_bhf = jnp.zeros_like(Bu)
        jacobians = {
            "nu_log": jnp.stack([
                dg_dnu * h_c1 - dphi_dnu * h_c2 + dnorm_dnu * u_c1,
                dg_dnu * h_c2 + dphi_dnu * h_c1 + dnorm_dnu * u_c2,
            ], axis=1),
            "theta_log": jnp.stack([
                dg_dtheta * h_c1 - dphi_dtheta * h_c2,
                dg_dtheta * h_c2 + dphi_dtheta * h_c1,
            ], axis=1),
            "B_real": jnp.stack([Bu, zeros_bhf], axis=1),
            "B_imag": jnp.stack([zeros_bhf, Bu], axis=1),
        }

        next_sensitivity = {}
        for name in sensitivity:
            S = sensitivity[name]
            J = jacobians[name]
            rotated = jnp.einsum('ijh,bjh...->bih...', A, S)
            next_sensitivity[name] = jnp.einsum('bih,bih...->bih...', d, rotated + J)

        return new_carry, new_carry, next_sensitivity

    def initialize_sensitivity(
        self, key: jax.Array, input_shape: Tuple[int, ...]
    ) -> Optional[Dict[str, Array]]:
        *batch_dims, _ = input_shape
        H = self.hidden_dim
        F = self.features
        return {
            "nu_log": jnp.zeros((*batch_dims, 2, H)),
            "theta_log": jnp.zeros((*batch_dims, 2, H)),
            "B_real": jnp.zeros((*batch_dims, 2, H, F)),
            "B_imag": jnp.zeros((*batch_dims, 2, H, F)),
        }
