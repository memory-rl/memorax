import jax.numpy as jnp  
import flax.linen as nn  

class CausalConv1D(nn.Module):  
    features: int  
    kernel_size: int  
    dilation: int = 1  
  
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: 
        """
        Args:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        
        Returns:
        - Convolved tensor of shape (batch_size, sequence_length, features)
        """
        assert x.ndim == 3, f"Input must be a 3D tensor (batch_size, sequence_length, features), got {x.ndim}D tensor"
        # Note: jax uses channels-last convention
        return nn.Conv(self.features, self.kernel_size, kernel_dilation=self.dilation)(x)
  
class BlockLinear(nn.Module):  
    out_features: int  
    num_blocks: int 
    use_bias: bool = True
  
    @nn.compact  
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args: 
        - x: Input tensor of shape (batch_size, in_features)
        
        Returns:
        - Output tensor of shape (batch_size, out_features)
        """
        assert x.ndim == 2, f"Input must be a 2D tensor (batch_size, in_features), but got {x.ndim}D tensor"
        
        block_out_features = self.out_features // self.num_blocks  
        x_split = jnp.split(x, self.num_blocks, axis=1)  
        y_split = [nn.Dense(block_out_features, use_bias=self.use_bias)(x_i) for x_i in x_split]
        return jnp.concatenate(y_split, axis=1)