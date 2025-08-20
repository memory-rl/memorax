from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import BlockLinear, CausalConv1D


@flax.struct.dataclass
class mLSTMCarry:
    C: jnp.ndarray
    n: jnp.ndarray
    x_prev: jnp.ndarray  # for 1D conv we need to store that past ker_size - 1 values
    m: jnp.ndarray


class mLSTM(nn.RNNCellBase):
    embedding_dim: int
    num_heads: int
    head_dim: int
    p_factor: int = 2
    use_bias: bool = True
    use_exp_f_gate = True
    ker_size: int = 4  # Default kernel size for 1D convolution
    use_conv: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @property
    def num_feature_axes(self) -> int:
        return 1

    @staticmethod
    def init_hidden(
        batch_size: int,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        ker_size: int,
        p_factor: float,
    ) -> mLSTMCarry:

        return mLSTMCarry(
            C=jnp.zeros(
                (batch_size, num_heads, head_dim, head_dim)
            ),  # (B, num_heads, head_dim, head_dim)
            n=jnp.ones((batch_size, num_heads, head_dim)),  # (B, num_heads, head_dim)
            x_prev=jnp.zeros(
                (batch_size, ker_size - 1, embedding_dim * p_factor)
            ),  # for 1D conv,
            m=jnp.zeros((batch_size, num_heads)),
        )

    def initialize_carry(
        self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]
    ) -> mLSTMCarry:
        batch_size = input_shape[0]  # assuming input_shape is (batch_size, ...)
        return mLSTM.init_hidden(
            batch_size=batch_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            ker_size=self.ker_size,
            p_factor=self.p_factor,
        )

    @nn.compact
    def __call__(
        self,
        carry: mLSTMCarry,
        inputs: jnp.ndarray,  # shape (batch_size, features)
    ) -> Tuple[
        mLSTMCarry,
        jnp.ndarray,  # shape (batch_size, features)
    ]:
        B, D = inputs.shape

        hid_dim = self.num_heads * self.head_dim
        # Init weights
        W_q = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name="W_q")
        W_k = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name="W_k")
        W_v = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name="W_v")
        W_i = nn.Dense(self.num_heads, use_bias=self.use_bias, name="W_i")
        W_f = nn.Dense(self.num_heads, use_bias=self.use_bias, name="W_f")
        W_o = nn.Dense(hid_dim, use_bias=self.use_bias, name="W_o")
        # skip = nn.Conv(hid_dim, kernel_size=1, use_bias=False)
        skip = nn.Dense(hid_dim, use_bias=False)

        group_norm = nn.GroupNorm(num_groups=self.num_heads)

        out_proj = nn.Dense(self.embedding_dim, use_bias=self.use_bias, name="out_proj")
        up_l_proj = nn.Dense(
            int(self.p_factor * self.embedding_dim),
            use_bias=self.use_bias,
            name="up_l_proj",
        )
        up_r_proj = nn.Dense(hid_dim, use_bias=self.use_bias, name="up_r_proj")

        # Apply weights
        x_n = nn.LayerNorm()(inputs)

        x_l = up_l_proj(x_n)  # (B, embedding_dim * p_factor)
        x_r = up_r_proj(x_n)  # (B, hid_dim)

        # x_prev is used to store the last ker_size - 1 values for causal convolution and has shape (batch_size, ker_size - 1, feature_dims)
        x_window = jnp.expand_dims(x_l, axis=1)  # shape (batch_size, 1, feature_dims)
        x_window = jnp.concatenate(
            [carry.x_prev, x_window], axis=1
        )  # shape (batch_size, ker_size, feature_dims)
        if self.use_conv:
            x_c = CausalConv1D(
                features=self.embedding_dim * self.p_factor, kernel_size=self.ker_size
            )(x_window)
            x_c = nn.silu(x_c)  # shape (batch_size, ker_size, feature_dims)
            x_c = x_c[:, -1, :]  # take the last value, shape (batch_size, feature_dims)
        else:
            x_c = x_l

        # update x_prev for the next step
        x_prev = x_window[:, 1:, :]  # shape (batch_size, ker_size - 1, feature_dims)

        # x_c = nn.silu(x_c)  # (B, embedding_dim * p_factor)

        q = W_q(x_c)
        k = W_k(x_c) / jnp.sqrt(self.head_dim)
        v = W_v(x_l)

        q = q.reshape(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)

        log_i = W_i(x_c)  # (B, num_heads)
        i = jnp.exp(log_i)  # (B, num_heads)
        pre_f = W_f(x_c)  # (B, num_heads)
        f = (
            jnp.exp(pre_f) if self.use_exp_f_gate else nn.sigmoid(pre_f)
        )  # (B, num_heads)
        log_f = pre_f if self.use_exp_f_gate else jnp.log(f)  # (B, num_heads)

        # o = jnp.exp(W_o(x_l))  # (B, hid_dim)
        o = nn.sigmoid(W_o(x_l)).reshape(B, self.num_heads, self.head_dim)

        # stabilize i and f
        m = jnp.maximum(log_f + carry.m, log_i)  # (B, num_heads) eq. 15
        i = jnp.exp(log_i - m)  # (B, num_heads) eq. 16
        f = jnp.exp(log_f + carry.m - m)  # (B, num_heads) eq. 17

        i = jnp.expand_dims(i, axis=2)  # (B, num_heads, 1)
        f = jnp.expand_dims(f, axis=2)  # (B, num_heads, 1)

        n = f * carry.n + i * k  # (B, num_heads, head_dim)

        i_expanded = jnp.expand_dims(i, axis=3)  # (B, num_heads, 1, 1)
        f_expanded = jnp.expand_dims(f, axis=3)  # (B, num_heads, 1, 1)
        C = f_expanded * carry.C + i_expanded * jnp.einsum(
            "bhv,bhk->bhvk", v, k
        )  # (B, num_heads, head_dim, head_dim)

        # h_denom = jnp.maximum(1, jnp.einsum("bhd,bhd->bh", n, q))  # (B, num_heads)
        # h_denom = jnp.expand_dims(h_denom, axis=2)  # (B, num_heads, 1)
        denom = jnp.einsum("bhd,bhd->bh", n, q)
        h_denom = jnp.maximum(1, abs(denom))
        h_denom = jnp.expand_dims(h_denom, axis=2)
        q_expanded = jnp.expand_dims(q, axis=3)  # (B, num_heads, head_dim, 1)
        assert q_expanded.shape == (B, self.num_heads, self.head_dim, 1)
        assert C.shape == (B, self.num_heads, self.head_dim, self.head_dim)
        h = C @ q_expanded  # (B, num_heads, head_dim, 1)
        h = jnp.squeeze(h, axis=3)  # (B, num_heads, head_dim)
        h = h / h_denom  # (B, num_heads, head_dim)

        # h_out = o * h.reshape(B, hid_dim)  # (B, hid_dim)
        h_out = (o * h).reshape(B, hid_dim)

        out = group_norm(h_out) + skip(x_c)  # (B, hid_dim)
        out = out * nn.silu(x_r)  # (B, hid_dim)
        out = out_proj(out)  # (B, embedding_dim)

        out = out + inputs

        return mLSTMCarry(C=C, n=n, x_prev=x_prev, m=m), out


if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embedding_dim = 64

    rnn = nn.RNN(
        mLSTM(
            embedding_dim=embedding_dim,
            num_heads=4,
            head_dim=16,
        )
    )

    x = jnp.ones((batch_size, seq_length, embedding_dim))
    variables = rnn.init(jax.random.PRNGKey(0), x)
    y = rnn.apply(variables, x)
    assert y.shape == (
        batch_size,
        seq_length,
        embedding_dim,
    ), f"Unexpected output shape: {y.shape}"
