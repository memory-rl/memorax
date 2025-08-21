import jax.numpy as jnp  
import flax.linen as nn  

class CausalConv1D(nn.Module):  
    features: int  
    kernel_size: int  
    dilation: int = 1  
    use_bias: bool = True
    depthwise: bool = True
  
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: 
        """
        Args:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        
        Returns:
        - Convolved tensor of shape (batch_size, sequence_length, features)
        """
        assert x.ndim == 3, f"Input must be a 3D tensor (batch_size, sequence_length, features), got {x.ndim}D tensor"

        B, T, C = x.shape

        if self.depthwise:
            assert self.features == C, f"Input features {C} does not match depthwise multiplier {self.features}"
            groups = C
        else:
            groups = 1

        left_padding = (self.kernel_size - 1) * self.dilation
        x = jnp.pad(x, ((0, 0), (left_padding, 0), (0, 0)))

        # Note: jax uses channels-last convention
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding="VALID",
            kernel_dilation=(self.dilation,),
            feature_group_count=groups,
            use_bias=self.use_bias,
        )(x)
  
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
