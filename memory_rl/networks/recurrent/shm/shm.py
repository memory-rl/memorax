from functools import partial
from typing import (
    Any,
    TypeVar,
)

import jax
from jax import numpy as jnp
from jax import random

from flax.linen import LayerNorm, initializers
from flax.linen.activation import sigmoid
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact, nowrap
from flax.typing import (
    Array,
    PRNGKey,
    Dtype,
    Initializer,
)
from flax.linen.recurrent import RNNCellBase

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class SHMCell(RNNCellBase):
    r"""Stable Hadamard Memory (SHM) cell.

    Following the Stable Hadamard Memory paper, the cell keeps a matrix memory
    :math:`M_t \in \mathbb{R}^{H \times H}`, reads with a query, and writes via
    element-wise (Hadamard) calibration plus a gated fast-weight style update:

    Read:
      .. math::
          h_t = M_t \, q(x_t)

    Write (Hadamard Memory Framework):
      .. math::
          M_t = M_{t-1} \odot C_\theta(x_t) + U_\phi(x_t)

    Update matrix (fast weights with a scalar gate):
      .. math::
          U_\phi(x_t) = \eta_\phi(x_t)\,[v(x_t) \otimes k(x_t)]

      where :math:`\eta_\phi(x_t) \in (0,1)` is a sigmoid gate and
      :math:`\otimes` is the outer product.

    Calibration matrix (SHM):
      .. math::
          C_\theta(x_t) = \mathbf{1} + \tanh\!\big(\,\theta_t \otimes v_c(x_t)\,\big)

      with :math:`\theta_t \in \mathbb{R}^H` chosen *per time step* by uniformly
      sampling a row from a learned table :math:`\Theta \in \mathbb{R}^{L\times H}`
      (to reduce timestep dependencies), and :math:`v_c:\mathbb{R}^D\to\mathbb{R}^H`
      implemented as a linear map.

    """

    features: int
    num_thetas: int = 128
    sample_theta: bool = True

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        """Applies SHM update and read for one step.

        Args:
          carry: matrix memory :math:`M_{t-1}` with shape ``(*batch, H, H)``.
          inputs: input :math:`x_t` with shape ``(*batch, D)``.

        Returns:
          (new_carry, output): where ``new_carry`` is :math:`M_t` with shape
          ``(*batch, H, H)`` and ``output`` is :math:`h_t` with shape ``(*batch, H)``.
        """
        M = carry
        H = self.features

        # Linear maps following paper sections:
        dense = partial(
            Dense,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        inputs = LayerNorm(
            epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype, name="ln"
        )(inputs)

        # v(x), k(x), q(x), v_c(x), and η(x) (gate) with sigmoid.
        v = dense(name="v", features=H)(inputs)  # (*, H)
        k = dense(name="k", features=H)(inputs)  # (*, H)
        q = dense(name="q", features=H)(inputs)  # (*, H)
        v_c = dense(name="vc", features=H)(inputs)  # (*, H)
        eta = sigmoid(dense(name="eta", features=1)(inputs))  # (*, 1)

        # U_t = eta * (v ⊗ k)  ∈ (*, H, H)
        U = (v[..., :, None] * k[..., None, :]) * eta[..., None]

        # Θ ∈ (L, H); choose θ_t ∈ (H,) per step. If sampling RNG not provided or
        # sample_theta=False, default to row 0 (deterministic).
        theta_table = self.param(
            "theta_table", self.kernel_init, (self.num_thetas, H), self.param_dtype
        )

        if self.sample_theta and self.has_rng("memory"):
            rng = self.make_rng(
                "memory"
            )  # split per step when used with RNN(split_rngs=...)

            idx = random.randint(rng, (), 0, self.num_thetas, dtype=jnp.int32)
            theta_t = theta_table[idx]
            theta_t = jnp.broadcast_to(theta_t, v_c.shape)  # (*, H)
        else:
            theta_t = jnp.broadcast_to(theta_table[0], v_c.shape)  # (*, H)

        # C_t = 1 + tanh( θ_t ⊗ v_c(x_t) )  ∈ (*, H, H)
        Ct = 1.0 + jnp.tanh(theta_t[..., :, None] * v_c[..., None, :])

        # Memory update: M_t = M_{t-1} ⊙ C_t + U_t
        M_new = M * Ct + U
        # jax.debug.print("M_new: {}", M_new)

        # Read: h_t = M_t q(x_t)
        h = jnp.einsum("...ij,...j->...i", M_new, q)

        return M_new, h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]) -> Array:
        """Initialize matrix memory M_0 as zeros with shape (*batch, H, H)."""
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features, self.features)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
