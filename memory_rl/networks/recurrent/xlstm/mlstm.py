from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import BlockLinear, CausalConv1D


@flax.struct.dataclass
class mLSTMStackCarry:
    C: jnp.ndarray      # (batch, num_layers, num_heads, head_dim, head_dim)
    n: jnp.ndarray      # (batch, num_layers, num_heads, head_dim)
    x_prev: jnp.ndarray # (batch, num_layers, ker_size-1, embedding_dim * p_factor)
    m: jnp.ndarray      # (batch, num_layers, num_heads)


class mLSTM(nn.RNNCellBase):
    embedding_dim: int
    num_heads: int
    head_dim: int
    num_layers: int = 1
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
        num_layers: int = 1,
    ) -> "mLSTMStackCarry":
        return mLSTMStackCarry(
            C=jnp.zeros((batch_size, num_layers, num_heads, head_dim, head_dim)),
            n=jnp.zeros((batch_size, num_layers, num_heads, head_dim)),
            x_prev=jnp.zeros((batch_size, num_layers, ker_size - 1, embedding_dim * p_factor)),
            m=jnp.zeros((batch_size, num_layers, num_heads)),
        )

    def initialize_carry(
        self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]
    ) -> mLSTMStackCarry:
        batch_size = input_shape[0]  # assuming input_shape is (batch_size, ...)
        return mLSTM.init_hidden(
            batch_size=batch_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            ker_size=self.ker_size,
            p_factor=self.p_factor,
            num_layers=self.num_layers,
        )

    @nn.compact
    def __call__(
        self,
        carry: "mLSTMStackCarry",
        inputs: jnp.ndarray,  # shape (batch_size, features)
    ) -> Tuple["mLSTMStackCarry", jnp.ndarray]:
        x = inputs
        B = x.shape[0]
        C_new = []
        n_new = []
        x_prev_new = []
        m_new = []
        for layer_idx in range(self.num_layers):
            hid_dim = self.num_heads * self.head_dim
            W_q = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name=f"W_q_{layer_idx}")
            W_k = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name=f"W_k_{layer_idx}")
            W_v = BlockLinear(hid_dim, self.num_heads, use_bias=self.use_bias, name=f"W_v_{layer_idx}")
            W_i = nn.Dense(self.num_heads, use_bias=self.use_bias, name=f"W_i_{layer_idx}")
            W_f = nn.Dense(self.num_heads, use_bias=self.use_bias, name=f"W_f_{layer_idx}")
            W_o = nn.Dense(hid_dim, use_bias=self.use_bias, name=f"W_o_{layer_idx}")
            skip = nn.Conv(hid_dim, kernel_size=1, use_bias=False, name=f"skip_{layer_idx}")

            group_norm = nn.GroupNorm(num_groups=self.num_heads, name=f"group_norm_{layer_idx}")

            out_proj = nn.Dense(self.embedding_dim, use_bias=self.use_bias, name=f"out_proj_{layer_idx}")
            up_l_proj = nn.Dense(
                int(self.p_factor * self.embedding_dim),
                use_bias=self.use_bias,
                name=f"up_l_proj_{layer_idx}",
            )
            up_r_proj = nn.Dense(hid_dim, use_bias=self.use_bias, name=f"up_r_proj_{layer_idx}")

            x_n = nn.LayerNorm(name=f"layer_norm_{layer_idx}")(x)
            x_l = up_l_proj(x_n)
            x_r = nn.silu(up_r_proj(x_n))

            x_window = jnp.expand_dims(x_l, axis=1)
            x_window = jnp.concatenate([carry.x_prev[:, layer_idx, ...], x_window], axis=1)
            if self.use_conv:
                x_c = CausalConv1D(
                    features=self.embedding_dim * self.p_factor, kernel_size=self.ker_size, name=f"causal_conv_{layer_idx}"
                )(x_window)
                x_c = nn.silu(x_c)
                x_c = x_c[:, -1, :]
            else:
                x_c = x_l

            x_prev = x_window[:, 1:, :]

            q = W_q(x_c)
            k = W_k(x_c) / jnp.sqrt(self.head_dim)
            v = W_v(x_l)

            q = q.reshape(B, self.num_heads, self.head_dim)
            k = k.reshape(B, self.num_heads, self.head_dim)
            v = v.reshape(B, self.num_heads, self.head_dim)

            log_i = W_i(x_c)
            i = jnp.exp(log_i)
            pre_f = W_f(x_c)
            f = (
                jnp.exp(pre_f) if self.use_exp_f_gate else nn.sigmoid(pre_f)
            )
            log_f = pre_f if self.use_exp_f_gate else jnp.log(f)

            o = nn.sigmoid(W_o(x_r)).reshape(B, self.num_heads, self.head_dim)

            m = jnp.maximum(log_f + carry.m[:, layer_idx, ...], log_i)
            i = jnp.exp(log_i - m)
            f = jnp.exp(log_f + carry.m[:, layer_idx, ...] - m)

            i = jnp.expand_dims(i, axis=2)
            f = jnp.expand_dims(f, axis=2)

            n = f * carry.n[:, layer_idx, ...] + i * k

            i_expanded = jnp.expand_dims(i, axis=3)
            f_expanded = jnp.expand_dims(f, axis=3)
            C = f_expanded * carry.C[:, layer_idx, ...] + i_expanded * jnp.einsum(
                "bhv,bhk->bhvk", v, k
            )

            denom = jnp.einsum("bhd,bhd->bh", n, q)
            h_denom = jnp.maximum(1, jnp.abs(denom))
            h_denom = jnp.expand_dims(h_denom, axis=2)
            q_expanded = jnp.expand_dims(q, axis=3)
            h = C @ q_expanded
            h = jnp.squeeze(h, axis=3)
            h = h / h_denom

            h_out = h.reshape(B, hid_dim)

            out = group_norm(h_out) + skip(x_c)
            out = (out.reshape(B, self.num_heads, self.head_dim) * o).reshape(B, hid_dim)
            out = out_proj(out)

            out = out + x

            C_new.append(C)
            n_new.append(n)
            x_prev_new.append(x_prev)
            m_new.append(m)
            x = out  # output of this layer is input to next

        # Stack along layer dimension, then transpose to batch-first
        new_carry = mLSTMStackCarry(
            C=jnp.stack(C_new, axis=1),        # (batch, num_layers, ...)
            n=jnp.stack(n_new, axis=1),
            x_prev=jnp.stack(x_prev_new, axis=1),
            m=jnp.stack(m_new, axis=1),
        )
        return new_carry, x


if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embedding_dim = 64

    rnn = nn.RNN(
        mLSTM(
            embedding_dim=embedding_dim,
            num_heads=4,
            head_dim=16,
            num_layers=2,  # Example: 2 layers
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
