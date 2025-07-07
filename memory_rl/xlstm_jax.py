import jax  
import jax.numpy as jnp  
import flax
import flax.linen as nn  
from typing import Tuple, Optional  



class CausalConv1D(nn.Module):  
    features: int  
    kernel_size: int  
    dilation: int = 1  
  
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: 
        """
        Args:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        
        Output:
        - Convolved tensor of shape (batch_size, sequence_length, features)
        """
        assert x.ndim == 3, f"Input must be a 3D tensor (batch_size, sequence_length, features), got {x.ndim}D tensor"
        # Note: jax uses channels-last convention
        return nn.Conv(self.features, self.kernel_size, kernel_dilation=self.dilation)(x)
  
  
class BlockLinear(nn.Module):  
    in_features: int  
    out_features: int  
    num_blocks: int  
  
    @nn.compact  
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args: 
        - x: Input tensor of shape (batch_size, in_features)
        Output:
        - Output tensor of shape (batch_size, out_features)
        """
        assert x.ndim == 2, f"Input must be a 2D tensor (batch_size, in_features), but got {x.ndim}D tensor"
        
        block_in_features = self.in_features // self.num_blocks  
        block_out_features = self.out_features // self.num_blocks  
        x_split = jnp.split(x, self.num_blocks, axis=1)  
        y_split = [nn.Dense(block_out_features)(x_i) for x_i in x_split]
        return jnp.concatenate(y_split, axis=1)
  
@flax.struct.dataclass
class sLSTMCarry():
    c: jnp.ndarray
    n: jnp.ndarray 
    h: jnp.ndarray 
    m: jnp.ndarray
    # for 1D conv we need to store that past ker_size - 1 values
    x_prev: jnp.ndarray 

class sLSTM(nn.Module):  
    inp_dim: int
    head_dim: int
    head_num: int
    ker_size: int = 4
    p_factor: float = 4 / 3
    eps: float = 1e-8 # for numerical stability
    use_conv: bool = False,
    
  
    @staticmethod
    def init_hidden(
        batch_size: int, inp_dim: int, head_num: int, head_dim: int, ker_size: int = 4
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: 

        return sLSTMCarry(
            c=jnp.zeros((batch_size, head_num * head_dim)),
            n=jnp.ones((batch_size, head_num * head_dim)),
            h=jnp.zeros((batch_size, head_num * head_dim)),
            m=jnp.zeros((batch_size, head_num * head_dim)),
            x_prev=jnp.zeros((batch_size, ker_size - 1, inp_dim))  # for 1D conv
        )
  
    @nn.compact
    def __call__(  
        self,
        carry: sLSTMCarry,
        inputs: jnp.ndarray, # shape (batch_size, features)
    ) -> Tuple[jnp.ndarray, # shape (batch_size, features)
               sLSTMCarry]:

        assert inputs.ndim == 2, f"Input must be a 2D tensor (batch_size, feature_dims), got {inputs.ndim}D tensor"
        assert inputs.shape[1] == self.inp_dim, f"Input feature dimension must be {self.inp_dim}, got {inputs.shape[1]}"
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

        R_z = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        R_i = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        R_f = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        R_o = BlockLinear(self.head_dim, self.head_dim, self.head_num)

        proj_dim = int(self.p_factor * self.head_num * self.head_dim)
        up_proj = nn.Dense(features=2 * proj_dim)
        down_proj = nn.Dense(features=self.inp_dim)

        c_tm1, n_tm1, h_tm1, m_tm1 = carry.c, carry.n, carry.h, carry.m
  
        x_t = inp_norm(inputs) # shape (batch_size, feature_dims)
        
        
        
        # x_prev is used to store the last ker_size - 1 values for causal convolution and has shape (batch_size, ker_size - 1, feature_dims)
        x_window = jnp.expand_dims(x_t, axis=1) # shape (batch_size, 1, feature_dims)
        x_window = jnp.concatenate([carry.x_prev, x_window], axis=1)  # shape (batch_size, ker_size, feature_dims)
        if self.use_conv:
            x_c = CausalConv1D(features=feature_dims, kernel_size=self.ker_size)(x_window)
            x_c = nn.silu(x_c) # shape (batch_size, ker_size, feature_dims)
            x_c = x_c[:, -1, :]  # take the last value, shape (batch_size, feature_dims)
        else:
            x_c = x_t
            
        # update x_prev for the next step
        x_prev = x_window[:, 1:, :]  # shape (batch_size, ker_size - 1, feature_dims)
  
  
        h_tm1_reshaped = h_tm1.reshape((batch_size, self.head_num, self.head_dim))  
          
        i_raw = W_i(x_c) + jax.vmap(R_i)(h_tm1_reshaped).reshape(batch_size, self.head_num * self.head_dim)  
        f_raw = W_f(x_c) + jax.vmap(R_f)(h_tm1_reshaped).reshape(batch_size, self.head_num * self.head_dim)  
        z_raw = W_z(x_c) + jax.vmap(R_z)(h_tm1_reshaped).reshape(batch_size, self.head_num * self.head_dim)  
        o_raw = W_o(x_c) + jax.vmap(R_o)(h_tm1_reshaped).reshape(batch_size, self.head_num * self.head_dim)  
    
        logfplusm = m_tm1 + jax.nn.log_sigmoid(f_raw)  
          
        # Handle the n == 0 case  
        m_t = jnp.where(  
            jnp.all(n_tm1 == 0.0, axis=-1, keepdims=True),  
            i_raw,  
            jnp.maximum(i_raw, logfplusm)  
        )  
          
        o_t = jax.nn.sigmoid(o_raw)  
        i_t = jnp.exp(i_raw - m_t)  
        f_t = jnp.exp(logfplusm - m_t)  
        z_t = jnp.tanh(z_raw)  
          
        c_t = f_t * c_tm1 + i_t * z_t  
        n_t = f_t * n_tm1 + i_t  
          
        # Compute hidden state with numerical stability protection  
        h_t = o_t * (c_t / (n_t + self.eps))  
  
        out = hid_norm(h_t.reshape(inputs.shape[0], self.head_num, self.head_dim)).reshape(  
            inputs.shape[0], -1  
        )  
  
        # GLU-style projection
        out1, out2 = jnp.split(up_proj(out), 2, axis=-1)  
        out = out1 * jax.nn.gelu(out2)  
        out = down_proj(out)  
        
        return sLSTMCarry(c=c_t, n=n_t, h=h_t, m=m_t, x_prev=x_prev), out + inputs
    
    
if __name__ == "__main__":  
    import jax  
    import jax.numpy as jnp  
    from jax import random  
      
    key = random.PRNGKey(42)  
      
    # Model configuration  
    batch_size = 2  
    seq_length = 10  
    inp_dim = 128  
    head_dim = 64  
    head_num = 4  
      
    # Initialize the model  
    model = sLSTM(  
        inp_dim=inp_dim,  
        head_dim=head_dim,  
        head_num=head_num,  
        ker_size=4,  
        p_factor=4/3,
        use_conv=False
    )  
      
    # Create dummy input data  
    # Shape: (batch_size, seq_length, inp_dim)  
    dummy_input = random.normal(key, (batch_size, inp_dim))  
      
    # Initialize model parameters  
    key, init_key = random.split(key)  

    initial_carry = sLSTM.init_hidden(batch_size=batch_size,
                                        inp_dim=inp_dim,
                                      head_num=head_num, 
                                      head_dim=head_dim)
    params = model.init(init_key, initial_carry, dummy_input)  

    print(f"Input shape: {dummy_input.shape}")
    print("\n--- Forward pass without convolution ---")  
    new_carry, output = model.apply(params, initial_carry, dummy_input)  
    print(f"Output shape: {output.shape}") 
      
    