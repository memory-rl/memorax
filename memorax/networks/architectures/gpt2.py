"""GPT-2 Architecture.

A decoder-only transformer following the GPT-2 design:
- Pre-norm (LayerNorm before attention and FFN)
- Learned absolute positional embeddings
- Causal self-attention
- GELU activation in FFN
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from memorax.networks.blocks import FFN, PreNorm, Residual
from memorax.networks.sequence_models.self_attention import Carry as AttentionCarry
from memorax.networks.sequence_models.self_attention import SelfAttention
from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import get_input_shape
from memorax.utils.typing import Array, Carry


@struct.dataclass
class GPT2Carry:
    position: Array
    attention_carries: tuple


class GPT2Block(SequenceModel):
    """Single GPT-2 transformer block.

    Combines self-attention with feed-forward network using
    pre-normalization and residual connections.

    Args:
        features: Model dimension.
        num_heads: Number of attention heads.
        context_length: Maximum sequence length for attention.
        expansion_factor: FFN hidden dimension multiplier.
        dropout_rate: Dropout probability.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
    """

    features: int
    num_heads: int
    context_length: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> AttentionCarry:
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads
        mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
        k = jnp.zeros(
            (batch_size, self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        v = jnp.zeros(
            (batch_size, self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return AttentionCarry(mask=mask, key=k, value=v)

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        attention = Residual(
            PreNorm(
                SelfAttention(
                    features=self.features,
                    num_heads=self.num_heads,
                    context_length=self.context_length,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
            )
        )

        ffn = Residual(
            PreNorm(
                FFN(
                    features=self.features,
                    expansion_factor=self.expansion_factor,
                    activation=nn.gelu,
                    dropout_rate=self.dropout_rate,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
            )
        )

        carry, x = attention(inputs, mask=mask, initial_carry=initial_carry, **kwargs)
        _, x = ffn(x, mask=mask, **kwargs)

        return carry, x


class GPT2(SequenceModel):
    """GPT-2 Architecture.

    A stack of transformer blocks with learned positional embeddings.

    Args:
        features: Model dimension (d_model).
        num_heads: Number of attention heads per layer.
        num_layers: Number of transformer blocks.
        context_length: Maximum sequence length.
        expansion_factor: FFN hidden dimension = features * expansion_factor.
        dropout_rate: Dropout probability.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.

    Example:
        >>> gpt2 = GPT2(
        ...     features=768,
        ...     num_heads=12,
        ...     num_layers=12,
        ...     context_length=1024,
        ... )
        >>> carry, outputs = gpt2(inputs, mask)
    """

    features: int
    num_heads: int
    num_layers: int
    context_length: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> GPT2Carry:
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads

        # Create attention carries for each layer
        attention_carries = tuple(
            AttentionCarry(
                mask=jnp.ones((batch_size, self.context_length), dtype=jnp.int32),
                key=jnp.zeros(
                    (batch_size, self.context_length, self.num_heads, head_dim),
                    dtype=self.dtype,
                ),
                value=jnp.zeros(
                    (batch_size, self.context_length, self.num_heads, head_dim),
                    dtype=self.dtype,
                ),
            )
            for _ in range(self.num_layers)
        )

        position = jnp.zeros((batch_size,), dtype=jnp.int32)
        return GPT2Carry(position=position, attention_carries=attention_carries)

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[GPT2Carry] = None,
        **kwargs,
    ) -> tuple[GPT2Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        batch_size, seq_len, *_ = inputs.shape

        # Learned positional embeddings
        position_embedding = nn.Embed(
            num_embeddings=self.context_length,
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="position_embedding",
        )

        # Compute positions
        time_indices = jnp.arange(seq_len)
        positions = initial_carry.position[:, None] + time_indices[None, :]
        positions = positions % self.context_length

        # Add positional embeddings
        x = inputs + position_embedding(positions)
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(x)

        # Process through transformer blocks
        new_carries = []
        for i in range(self.num_layers):
            block = GPT2Block(
                features=self.features,
                num_heads=self.num_heads,
                context_length=self.context_length,
                expansion_factor=self.expansion_factor,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"block_{i}",
            )
            carry, x = block(
                x, mask=mask, initial_carry=initial_carry.attention_carries[i], **kwargs
            )
            new_carries.append(carry)

        # Final layer norm
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln_f")(x)

        # Update position
        new_position = initial_carry.position + seq_len

        return GPT2Carry(position=new_position, attention_carries=tuple(new_carries)), x
