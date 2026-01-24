"""Transformer-XL Architecture.

A transformer with segment-level recurrence for modeling long sequences:
- Segment recurrence: cache previous segment outputs as memory
- Relative positional embeddings for cross-segment attention
- Pre-norm transformer blocks
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from memorax.networks.blocks import FFN, PreNorm, Residual, SegmentRecurrence
from memorax.networks.blocks.segment_recurrence import Memory
from memorax.networks.positional_embeddings.rope import RoPE
from memorax.networks.sequence_models.self_attention import Carry as AttentionCarry
from memorax.networks.sequence_models.self_attention import SelfAttention
from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import get_input_shape
from memorax.utils.typing import Array, Carry


class TransformerXLBlock(SequenceModel):
    """Single Transformer-XL block with segment recurrence.

    Combines self-attention (with cross-segment memory) with FFN.
    Uses relative positional embeddings for position-aware attention
    across segment boundaries.

    Args:
        features: Model dimension.
        num_heads: Number of attention heads.
        context_length: Maximum sequence length within a segment.
        memory_length: Number of past timesteps to cache for cross-segment attention.
        expansion_factor: FFN hidden dimension multiplier.
        dropout_rate: Dropout probability.
        positional_embedding: Relative positional embedding function.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
    """

    features: int
    num_heads: int
    context_length: int
    memory_length: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0
    positional_embedding: Callable = RoPE()
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> tuple[Memory, AttentionCarry]:
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads

        # Memory for segment recurrence (pre-allocated with zeros)
        memory = Memory(
            state=jnp.zeros((batch_size, self.memory_length, self.features), dtype=self.dtype),
            mask=jnp.zeros((batch_size, self.memory_length), dtype=jnp.int32),
        )

        # Attention carry
        attention_carry = AttentionCarry(
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

        return (memory, attention_carry)

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
                SegmentRecurrence(
                    sequence_model=SelfAttention(
                        features=self.features,
                        num_heads=self.num_heads,
                        context_length=self.context_length,
                        positional_embedding=self.positional_embedding,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                    ),
                    memory_length=self.memory_length,
                    features=self.features,
                    dtype=self.dtype,
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


@struct.dataclass
class TransformerXLCarry:
    block_carries: tuple


class TransformerXL(SequenceModel):
    """Transformer-XL Architecture.

    A transformer with segment-level recurrence that enables modeling
    of sequences longer than the context window by caching and attending
    to previous segment outputs.

    Key features:
    - Segment recurrence: each layer caches its outputs for cross-segment attention
    - Relative positional embeddings (RoPE by default) for position-aware attention
    - Pre-norm architecture with residual connections

    Args:
        features: Model dimension (d_model).
        num_heads: Number of attention heads per layer.
        num_layers: Number of transformer blocks.
        context_length: Maximum sequence length within a segment.
        memory_length: Number of past timesteps cached per layer.
        expansion_factor: FFN hidden dimension = features * expansion_factor.
        dropout_rate: Dropout probability.
        positional_embedding: Relative positional embedding function.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.

    Example:
        >>> trxl = TransformerXL(
        ...     features=512,
        ...     num_heads=8,
        ...     num_layers=6,
        ...     context_length=512,
        ...     memory_length=512,
        ... )
        >>> # Process sequence in segments
        >>> carry, outputs = trxl(segment_1, mask)
        >>> carry, outputs = trxl(segment_2, mask, initial_carry=carry)
    """

    features: int
    num_heads: int
    num_layers: int
    context_length: int
    memory_length: int
    expansion_factor: int = 4
    dropout_rate: float = 0.0
    positional_embedding: Callable = RoPE()
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> TransformerXLCarry:
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads

        # Create carries for each block
        block_carries = tuple(
            (
                Memory(
                    state=jnp.zeros((batch_size, self.memory_length, self.features), dtype=self.dtype),
                    mask=jnp.zeros((batch_size, self.memory_length), dtype=jnp.int32),
                ),
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
                ),
            )
            for _ in range(self.num_layers)
        )
        return TransformerXLCarry(block_carries=block_carries)

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[TransformerXLCarry] = None,
        **kwargs,
    ) -> tuple[TransformerXLCarry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        # Input projection if needed
        if inputs.shape[-1] != self.features:
            x = nn.Dense(
                self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name="input_projection",
            )(inputs)
        else:
            x = inputs

        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(x)

        # Process through transformer blocks
        new_carries = []
        for i in range(self.num_layers):
            block = TransformerXLBlock(
                features=self.features,
                num_heads=self.num_heads,
                context_length=self.context_length,
                memory_length=self.memory_length,
                expansion_factor=self.expansion_factor,
                dropout_rate=self.dropout_rate,
                positional_embedding=self.positional_embedding,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"block_{i}",
            )
            carry, x = block(
                x, mask=mask, initial_carry=initial_carry.block_carries[i], **kwargs
            )
            new_carries.append(carry)

        # Final layer norm
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln_f")(x)

        return TransformerXLCarry(block_carries=tuple(new_carries)), x
