from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import BlockLinear, CausalConv1D


@flax.struct.dataclass
class sLSTMCarry:
    c: jnp.ndarray
    n: jnp.ndarray
    h: jnp.ndarray
    m: jnp.ndarray
    x_prev: jnp.ndarray  # for 1D conv we need to store that past ker_size - 1 values


class sLSTM(nn.RNNCellBase):
    inp_dim: int
    head_dim: int
    head_num: int
    ker_size: int = 4  # in almost all cases ker_size should be 4
    p_factor: float = 4 / 3
    eps: float = 1e-8  # for numerical stability
    use_conv: bool = (False,)

    @property
    def num_feature_axes(self) -> int:
        return 1

    @staticmethod
    def init_hidden(
        batch_size: int, inp_dim: int, head_num: int, head_dim: int, ker_size: int = 4
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        return sLSTMCarry(
            c=jnp.zeros((batch_size, head_num * head_dim)),
            n=jnp.ones((batch_size, head_num * head_dim)),
            h=jnp.zeros((batch_size, head_num * head_dim)),
            m=jnp.zeros((batch_size, head_num * head_dim)),
            x_prev=jnp.zeros((batch_size, ker_size - 1, inp_dim)),  # for 1D conv
        )

    def initialize_carry(
        self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]
    ) -> sLSTMCarry:
        batch_size = input_shape[0]  # assuming input_shape is (batch_size, ...)
        return sLSTM.init_hidden(
            batch_size=batch_size,
            inp_dim=self.inp_dim,
            head_num=self.head_num,
            head_dim=self.head_dim,
            ker_size=self.ker_size,
        )

    @nn.compact
    def __call__(
        self,
        carry: sLSTMCarry,
        inputs: jnp.ndarray,  # shape (batch_size, features)
    ) -> Tuple[jnp.ndarray, sLSTMCarry]:  # shape (batch_size, features)

        assert (
            inputs.ndim == 2
        ), f"Input must be a 2D tensor (batch_size, feature_dims), got {inputs.ndim}D tensor"
        assert (
            inputs.shape[1] == self.inp_dim
        ), f"Input feature dimension must be {self.inp_dim}, got {inputs.shape[1]}"
        assert self.head_num > 0, "head_num must be greater than 0"
        assert self.head_dim > 0, "head_dim must be greater than 0"
        assert self.p_factor > 0, "p_factor must be greater than 0"

        batch_size, feature_dims = inputs.shape

        inp_norm = nn.LayerNorm(self.inp_dim)
        hid_norm = nn.GroupNorm(num_groups=self.head_num)

        W_z = nn.Dense(features=self.head_num * self.head_dim)
        W_i = nn.Dense(features=self.head_num * self.head_dim)
        W_o = nn.Dense(features=self.head_num * self.head_dim)
        W_f = nn.Dense(features=self.head_num * self.head_dim)

        R_z = BlockLinear(
            out_features=self.head_num * self.head_dim, num_blocks=self.head_num
        )
        R_i = BlockLinear(
            out_features=self.head_num * self.head_dim, num_blocks=self.head_num
        )
        R_f = BlockLinear(
            out_features=self.head_num * self.head_dim, num_blocks=self.head_num
        )
        R_o = BlockLinear(
            out_features=self.head_num * self.head_dim, num_blocks=self.head_num
        )

        proj_dim = int(self.p_factor * self.head_num * self.head_dim)
        up_proj = nn.Dense(features=2 * proj_dim)
        down_proj = nn.Dense(features=self.inp_dim)

        c_tm1, n_tm1, h_tm1, m_tm1 = carry.c, carry.n, carry.h, carry.m

        x_t = inp_norm(inputs)  # shape (batch_size, feature_dims)

        # x_prev is used to store the last ker_size - 1 values for causal convolution and has shape (batch_size, ker_size - 1, feature_dims)
        x_window = jnp.expand_dims(x_t, axis=1)  # shape (batch_size, 1, feature_dims)
        x_window = jnp.concatenate(
            [carry.x_prev, x_window], axis=1
        )  # shape (batch_size, ker_size, feature_dims)
        if self.use_conv:
            x_c = CausalConv1D(features=feature_dims, kernel_size=self.ker_size)(
                x_window
            )
            x_c = nn.silu(x_c)  # shape (batch_size, ker_size, feature_dims)
            x_c = x_c[:, -1, :]  # take the last value, shape (batch_size, feature_dims)
        else:
            x_c = x_t

        # update x_prev for the next step
        x_prev = x_window[:, 1:, :]  # shape (batch_size, ker_size - 1, feature_dims)

        i_raw = W_i(x_c) + R_i(h_tm1)
        f_raw = W_f(x_c) + R_f(h_tm1)
        z_raw = W_z(x_c) + R_z(h_tm1)
        o_raw = W_o(x_c) + R_o(h_tm1)

        logfplusm = m_tm1 + jax.nn.log_sigmoid(f_raw)

        # Handle the n == 0 case
        m_t = jnp.where(
            jnp.all(n_tm1 == 0.0, axis=-1, keepdims=True),
            i_raw,
            jnp.maximum(i_raw, logfplusm),
        )

        o_t = jax.nn.sigmoid(o_raw)
        i_t = jnp.exp(i_raw - m_t)
        f_t = jnp.exp(logfplusm - m_t)
        z_t = jnp.tanh(z_raw)

        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t

        # Compute hidden state with numerical stability protection
        h_t = o_t * (c_t / (n_t + self.eps))

        out = hid_norm(
            h_t.reshape(inputs.shape[0], self.head_num, self.head_dim)
        ).reshape(inputs.shape[0], -1)

        # GLU-style projection
        out1, out2 = jnp.split(up_proj(out), 2, axis=-1)
        out = out1 * jax.nn.gelu(out2)
        out = down_proj(out)

        return sLSTMCarry(c=c_t, n=n_t, h=h_t, m=m_t, x_prev=x_prev), out + inputs


if __name__ == "__main__":
    # Example usage of sLSTM with RNN

    batch_size = 2
    seq_length = 10
    inp_dim = 32

    rnn = nn.RNN(
        sLSTM(
            inp_dim=inp_dim,
            head_dim=64,
            head_num=4,
            ker_size=4,  # in almost all cases ker_size should be 4
            p_factor=4 / 3,  # p_factor of 4/3 as in xLSTM paper
            eps=1e-8,  # for numerical stability
            use_conv=True,
        )
    )

    x = jnp.ones((batch_size, seq_length, inp_dim))
    variables = rnn.init(jax.random.PRNGKey(0), x)
    y = rnn.apply(variables, x)
    print(f"Output shape of rnn: {y.shape}")
