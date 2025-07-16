import jax  
import jax.numpy as jnp  
import flax
import flax.linen as nn  
from typing import Tuple


@flax.struct.dataclass
class mLSTMCarry():
    C: jnp.ndarray
    n: jnp.ndarray
    x_prev: jnp.ndarray # for 1D conv we need to store that past ker_size - 1 values

# TODO: this is currently only the inner part of mLSTM, not the full model
# Full model reference: https://arxiv.org/pdf/2405.04517 page 30
# will add this tomorrow
class mLSTM(nn.RNNCellBase):  
    num_heads: int
    embedding_dim: int
    v_dim_factor: float
    qk_dim_factor: float
    use_bias: bool = True
    use_exp_f_gate = True
    ker_size: int = 4  # Default kernel size for 1D convolution
    
    @property
    def num_feature_axes(self) -> int:
        return 1
    
    @staticmethod
    def init_hidden(
        batch_size: int, 
        embedding_dim: int, 
        num_heads: int, 
        v_dim_factor: float,
        qk_dim_factor: float,
        ker_size: int
    ) -> mLSTMCarry:

        v_dim = int(embedding_dim * v_dim_factor)
        qk_dim = int(embedding_dim * qk_dim_factor)
        head_dim_qk = qk_dim // num_heads
        head_dim_v = v_dim // num_heads
        
        return mLSTMCarry(
            C=jnp.zeros((batch_size, num_heads, head_dim_v, head_dim_qk)), # (B, num_heads, head_dim_v, head_dim_qk)
            n=jnp.ones((batch_size, num_heads, head_dim_qk)), # (B, num_heads, head_dim_qk)
            x_prev=jnp.zeros((batch_size, ker_size - 1, embedding_dim))  # for 1D conv
        )
  
    def initialize_carry(self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]) -> mLSTMCarry:
        batch_size = input_shape[0]  # assuming input_shape is (batch_size, ...)
        return mLSTM.init_hidden(
            batch_size=batch_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            v_dim_factor=self.v_dim_factor,
            qk_dim_factor=self.qk_dim_factor,
            ker_size=self.ker_size
        )
  
    @nn.compact
    def __call__(  
        self,
        carry: mLSTMCarry,
        inputs: jnp.ndarray, # shape (batch_size, features)
    ) -> Tuple[mLSTMCarry,
        jnp.ndarray, # shape (batch_size, features)
            ]:
        # Calculate dimensions and assert input shapes
        v_dim = int(self.embedding_dim * self.v_dim_factor)
        qk_dim = int(self.embedding_dim * self.qk_dim_factor)
        head_dim_qk = qk_dim // self.num_heads
        head_dim_v = v_dim // self.num_heads
        
        assert inputs.ndim == 2, f"Input must have shape (B, D), got {inputs.shape}"
        B, D = inputs.shape
        
        assert carry.C.shape == (B, self.num_heads, head_dim_v, head_dim_qk), \
            f"Carry C must have shape (B, num_heads, head_dim_v, head_dim_qk), got {carry.C.shape}"
        assert carry.n.shape == (B, self.num_heads, head_dim_qk), \
            f"Carry n must have shape (B, num_heads, head_dim_qk), got {carry.n.shape}"
        assert carry.x_prev.shape == (B, self.ker_size - 1, D), \
            f"Carry x_prev must have shape (B, ker_size - 1, D), got {carry.x_prev.shape}"

        # Init weights        
        W_q = nn.Dense(qk_dim, use_bias=self.use_bias, name='W_q')
        W_k = nn.Dense(qk_dim, use_bias=self.use_bias, name='W_k')
        W_v = nn.Dense(v_dim, use_bias=self.use_bias, name='W_v')
        W_i = nn.Dense(self.num_heads, use_bias=self.use_bias, name='W_i')
        W_f = nn.Dense(self.num_heads, use_bias=self.use_bias, name='W_f')
        W_o = nn.Dense(v_dim, use_bias=self.use_bias, name='W_o')
        
        out_proj = nn.Dense(self.embedding_dim, use_bias=self.use_bias, name='out_proj')
        
        # Apply weights
        q = W_q(inputs)
        k = W_k(inputs) / jnp.sqrt(qk_dim)
        v = W_v(inputs)

        q = q.reshape(B, self.num_heads, head_dim_qk)  # (B, num_heads, head_dim_qk)
        k = k.reshape(B, self.num_heads, head_dim_qk)  # (B, num_heads, head_dim_qk)
        v = v.reshape(B, self.num_heads, head_dim_v)   # (B, num_heads, head_dim_v)

        i = jnp.exp(W_i(inputs))
        f = jnp.exp(W_f(inputs)) if self.use_exp_f_gate else nn.sigmoid(W_f(inputs))
        o = jnp.exp(W_o(inputs))
        o = o.reshape(B, self.num_heads, head_dim_v)   # (B, num_heads, head_dim_v)
        
        i = jnp.expand_dims(i, axis=2)  # (B, num_heads, 1)
        f = jnp.expand_dims(f, axis=2)  # (B, num_heads, 1)
        
        n = f * carry.n + i * k # (B, num_heads, head_dim_qk)
        
        i_expanded = jnp.expand_dims(i, axis=3)  # (B, num_heads, 1, 1)
        f_expanded = jnp.expand_dims(f, axis=3)  # (B, num_heads, 1, 1)
        C = f_expanded * carry.C + i_expanded * jnp.einsum('bhv,bhk->bhvk', v, k) # (B, num_heads, head_dim_v, head_dim_qk)
        
        h_denom = jnp.maximum(1, jnp.einsum('bhd,bhd->bh', n, q)) # (B, num_heads)
        h_denom = jnp.expand_dims(h_denom, axis=2) # (B, num_heads, 1)
        q_expanded = jnp.expand_dims(q, axis=3)  # (B, num_heads, head_dim_qk, 1)
        assert q_expanded.shape == (B, self.num_heads, head_dim_qk, 1)
        assert C.shape == (B, self.num_heads, head_dim_v, head_dim_qk)
        h = C @ q_expanded # (B, num_heads, head_dim_v, 1)
        h = jnp.squeeze(h, axis=3) # (B, num_heads, head_dim_v)
        h = h / h_denom # (B, num_heads, head_dim_v)
        
        h_out = o * h # (B, num_heads, head_dim_v)
        
        h_out = h_out.reshape(B, -1)
        h_out = out_proj(h_out)  # (B, embedding_dim)
        
        return mLSTMCarry(
            C=C,
            n=n,
            x_prev=carry.x_prev # TODO: handle x_prev if needed in mLSTMCarry
        ), h_out
        


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    embedding_dim = 64
    v_dim_factor = 1.0
    qk_dim_factor = 1.0
    num_heads = 4
    ker_size = 4

    m_lstm = mLSTM(
        embedding_dim=embedding_dim,
        v_dim_factor=v_dim_factor,
        qk_dim_factor=qk_dim_factor,
        num_heads=num_heads,
        ker_size=ker_size
    )

    rng = jax.random.PRNGKey(0)
    inputs = jax.random.normal(rng, (batch_size, embedding_dim))
    
    carry = m_lstm.initialize_carry(rng, inputs.shape)
    
    output, new_carry = m_lstm(carry, inputs)
    
    print("Output shape:", output.shape)
    print("New carry:", new_carry)