# Copyright 2024 Memorax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas kernels for linear attention.

Implements fused kernels for linear attention computation following the
memoroid algebra pattern. These kernels optimize:
1. Key-value outer product computation (write)
2. Query-based memory retrieval with normalization (read)

Based on "Transformers are RNNs" (Katharopoulos et al., 2020) with
optimizations inspired by FlashLinearAttention patterns.
"""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

# Default epsilon for numerical stability
DEFAULT_EPS = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class LinearAttentionBlockSizes:
    """Block sizes for linear attention kernels.

    Attributes:
        block_seq: Block size along sequence dimension for forward pass.
        block_head_dim: Block size along head dimension.
        block_seq_bwd: Block size along sequence for backward pass.
    """

    block_seq: int = 64
    block_head_dim: int = 64
    block_seq_bwd: int = 32

    @classmethod
    def get_default(cls) -> "LinearAttentionBlockSizes":
        return cls()

    @classmethod
    def for_head_dim(cls, head_dim: int) -> "LinearAttentionBlockSizes":
        """Get block sizes tuned for a specific head dimension."""
        if head_dim <= 32:
            return cls(block_seq=128, block_head_dim=32, block_seq_bwd=64)
        elif head_dim <= 64:
            return cls(block_seq=64, block_head_dim=64, block_seq_bwd=32)
        else:
            return cls(block_seq=32, block_head_dim=head_dim, block_seq_bwd=16)


def _elu_plus_one(x: jax.Array) -> jax.Array:
    """ELU+1 feature map: elu(x) + 1.

    This ensures non-negative outputs for proper normalization.
    """
    return jnp.where(x > 0, x + 1, jnp.exp(x))


def _feature_map_kernel(
    x_ref,
    out_ref,
    *,
    block_seq: int,
    block_dim: int,
):
    """Pallas kernel for applying feature map to inputs.

    Applies ELU+1 feature map in a block-tiled fashion.
    """
    seq_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    batch_idx = pl.program_id(2)

    seq_slice = pl.dslice(seq_idx * block_seq, block_seq)
    dim_slice = pl.dslice(0, block_dim)

    # Load block
    x = pl.load(x_ref, (batch_idx, seq_slice, head_idx, dim_slice))

    # Apply ELU+1 feature map
    out = jnp.where(x > 0, x + 1, jnp.exp(x))

    # Store result
    pl.store(out_ref, (batch_idx, seq_slice, head_idx, dim_slice), out)


def _outer_product_kernel(
    key_ref,
    value_ref,
    state_ref,
    normalizer_ref,
    *,
    block_seq: int,
    head_dim: int,
):
    """Pallas kernel for computing key-value outer products.

    Computes:
        state[t] = value[t] ⊗ key[t]
        normalizer[t] = key[t]

    This is the "write" operation in linear attention.
    """
    seq_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    batch_idx = pl.program_id(2)

    seq_slice = pl.dslice(seq_idx * block_seq, block_seq)
    dim_slice = pl.dslice(0, head_dim)

    # Load key and value blocks
    key = pl.load(key_ref, (batch_idx, seq_slice, head_idx, dim_slice))
    value = pl.load(value_ref, (batch_idx, seq_slice, head_idx, dim_slice))

    # Compute outer product for each position in block
    # state[b, t, h, i, j] = value[b, t, h, i] * key[b, t, h, j]
    state = jnp.einsum("ti,tj->tij", value, key)

    # Store results
    pl.store(
        state_ref,
        (batch_idx, seq_slice, head_idx, pl.dslice(0, head_dim), dim_slice),
        state,
    )
    pl.store(normalizer_ref, (batch_idx, seq_slice, head_idx, dim_slice), key)


def _fused_kv_outer_product_kernel(
    x_ref,
    key_weight_ref,
    value_weight_ref,
    state_ref,
    normalizer_ref,
    *,
    block_seq: int,
    head_dim: int,
    in_features: int,
):
    """Fused kernel for projection + feature map + outer product.

    Performs in a single kernel:
    1. Project input to keys and values
    2. Apply ELU+1 feature map to keys
    3. Compute outer product state = v ⊗ φ(k)
    """
    seq_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    batch_idx = pl.program_id(2)

    seq_slice = pl.dslice(seq_idx * block_seq, block_seq)

    # Load input block
    x = pl.load(x_ref, (batch_idx, seq_slice, pl.dslice(0, in_features)))

    # Load weight slices for this head
    key_w = pl.load(
        key_weight_ref, (pl.dslice(0, in_features), head_idx, pl.dslice(0, head_dim))
    )
    value_w = pl.load(
        value_weight_ref, (pl.dslice(0, in_features), head_idx, pl.dslice(0, head_dim))
    )

    # Project: (block_seq, in_features) @ (in_features, head_dim) -> (block_seq, head_dim)
    key = pl.dot(x, key_w)
    value = pl.dot(x, value_w)

    # Apply feature map to keys
    key = jnp.where(key > 0, key + 1, jnp.exp(key))

    # Compute outer product
    state = jnp.einsum("ti,tj->tij", value, key)

    # Store
    pl.store(
        state_ref,
        (batch_idx, seq_slice, head_idx, pl.dslice(0, head_dim), pl.dslice(0, head_dim)),
        state,
    )
    pl.store(normalizer_ref, (batch_idx, seq_slice, head_idx, pl.dslice(0, head_dim)), key)


def _read_kernel(
    state_ref,
    normalizer_ref,
    query_ref,
    out_ref,
    *,
    block_seq: int,
    head_dim: int,
    eps: float,
):
    """Pallas kernel for reading from accumulated state.

    Computes normalized linear attention output:
        numerator = φ(q) @ S = Σ (φ(q) · φ(k)) * v
        denominator = φ(q) @ z = Σ φ(q) · φ(k)
        output = numerator / max(denominator, eps)
    """
    seq_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    batch_idx = pl.program_id(2)

    seq_slice = pl.dslice(seq_idx * block_seq, block_seq)
    dim_slice = pl.dslice(0, head_dim)

    # Load query (already has feature map applied)
    query = pl.load(query_ref, (batch_idx, seq_slice, head_idx, dim_slice))

    # Load accumulated state and normalizer
    state = pl.load(
        state_ref,
        (batch_idx, seq_slice, head_idx, dim_slice, dim_slice),
    )
    normalizer = pl.load(normalizer_ref, (batch_idx, seq_slice, head_idx, dim_slice))

    # Numerator: query @ state -> (block_seq, head_dim)
    # state is (block_seq, head_dim, head_dim)
    # query is (block_seq, head_dim)
    numerator = jnp.einsum("tij,tj->ti", state, query)

    # Denominator: dot product of query and normalizer
    denominator = jnp.sum(query * normalizer, axis=-1, keepdims=True)
    denominator = jnp.maximum(denominator, eps)

    # Normalized output
    output = numerator / denominator

    pl.store(out_ref, (batch_idx, seq_slice, head_idx, dim_slice), output)


def _fused_read_kernel(
    state_ref,
    normalizer_ref,
    x_ref,
    query_weight_ref,
    out_ref,
    *,
    block_seq: int,
    head_dim: int,
    in_features: int,
    eps: float,
):
    """Fused kernel for query projection + feature map + read.

    Performs in a single kernel:
    1. Project input to queries
    2. Apply ELU+1 feature map
    3. Compute normalized output from accumulated state
    """
    seq_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    batch_idx = pl.program_id(2)

    seq_slice = pl.dslice(seq_idx * block_seq, block_seq)

    # Load input and project to query
    x = pl.load(x_ref, (batch_idx, seq_slice, pl.dslice(0, in_features)))
    query_w = pl.load(
        query_weight_ref,
        (pl.dslice(0, in_features), head_idx, pl.dslice(0, head_dim)),
    )
    query = pl.dot(x, query_w)

    # Apply feature map
    query = jnp.where(query > 0, query + 1, jnp.exp(query))

    # Load accumulated state and normalizer
    state = pl.load(
        state_ref,
        (
            batch_idx,
            seq_slice,
            head_idx,
            pl.dslice(0, head_dim),
            pl.dslice(0, head_dim),
        ),
    )
    normalizer = pl.load(
        normalizer_ref, (batch_idx, seq_slice, head_idx, pl.dslice(0, head_dim))
    )

    # Compute normalized output
    numerator = jnp.einsum("tij,tj->ti", state, query)
    denominator = jnp.sum(query * normalizer, axis=-1, keepdims=True)
    denominator = jnp.maximum(denominator, eps)
    output = numerator / denominator

    pl.store(out_ref, (batch_idx, seq_slice, head_idx, pl.dslice(0, head_dim)), output)


def _causal_linear_attention_kernel(
    query_ref,
    key_ref,
    value_ref,
    out_ref,
    state_ref,
    normalizer_ref,
    *,
    block_seq: int,
    head_dim: int,
    eps: float,
):
    """Fused causal linear attention kernel.

    Computes causal linear attention in a single pass using recurrent form:
        S_t = S_{t-1} + v_t ⊗ φ(k_t)
        z_t = z_{t-1} + φ(k_t)
        o_t = (φ(q_t) @ S_t) / (φ(q_t) · z_t)

    This kernel processes blocks sequentially, maintaining running state.
    """
    batch_idx = pl.program_id(0)
    head_idx = pl.program_id(1)
    seq_len = query_ref.shape[1]

    dim_slice = pl.dslice(0, head_dim)

    # Initialize running state
    running_state = jnp.zeros((head_dim, head_dim), dtype=jnp.float32)
    running_normalizer = jnp.zeros((head_dim,), dtype=jnp.float32)

    def body_fn(t, carry):
        running_state, running_normalizer = carry

        # Load single position
        query = pl.load(query_ref, (batch_idx, t, head_idx, dim_slice))
        key = pl.load(key_ref, (batch_idx, t, head_idx, dim_slice))
        value = pl.load(value_ref, (batch_idx, t, head_idx, dim_slice))

        # Apply feature map to query and key
        query = jnp.where(query > 0, query + 1, jnp.exp(query))
        key = jnp.where(key > 0, key + 1, jnp.exp(key))

        # Update running state: S_t = S_{t-1} + v_t ⊗ k_t
        running_state = running_state + jnp.outer(value, key)
        running_normalizer = running_normalizer + key

        # Compute output: o_t = (q_t @ S_t) / (q_t · z_t)
        numerator = running_state @ query
        denominator = jnp.maximum(jnp.dot(query, running_normalizer), eps)
        output = numerator / denominator

        pl.store(out_ref, (batch_idx, t, head_idx, dim_slice), output)

        return running_state, running_normalizer

    final_state, final_normalizer = lax.fori_loop(
        0, seq_len, body_fn, (running_state, running_normalizer)
    )

    # Store final state for potential continuation
    pl.store(
        state_ref,
        (batch_idx, head_idx, dim_slice, dim_slice),
        final_state,
    )
    pl.store(normalizer_ref, (batch_idx, head_idx, dim_slice), final_normalizer)


def compute_kv_state(
    key: jax.Array,
    value: jax.Array,
    *,
    block_sizes: LinearAttentionBlockSizes | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Compute key-value outer product state using Pallas.

    Args:
        key: Keys with feature map applied, shape (B, T, H, D)
        value: Values, shape (B, T, H, D)
        block_sizes: Optional block sizes configuration
        interpret: Whether to run in interpret mode (for debugging)
        debug: Enable debug output

    Returns:
        Tuple of (state, normalizer) where:
            state: Shape (B, T, H, D, D)
            normalizer: Shape (B, T, H, D)
    """
    if block_sizes is None:
        block_sizes = LinearAttentionBlockSizes.get_default()

    batch_size, seq_len, num_heads, head_dim = key.shape
    block_seq = min(block_sizes.block_seq, seq_len)

    if seq_len % block_seq != 0:
        # Pad sequence to block size
        pad_len = block_seq - (seq_len % block_seq)
        key = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        padded_seq_len = seq_len + pad_len
    else:
        padded_seq_len = seq_len
        pad_len = 0

    grid = (padded_seq_len // block_seq, num_heads, batch_size)

    out_shapes = [
        jax.ShapeDtypeStruct(
            (batch_size, padded_seq_len, num_heads, head_dim, head_dim), key.dtype
        ),
        jax.ShapeDtypeStruct(
            (batch_size, padded_seq_len, num_heads, head_dim), key.dtype
        ),
    ]

    kernel = functools.partial(
        _outer_product_kernel,
        block_seq=block_seq,
        head_dim=head_dim,
    )

    in_specs = [
        pl.BlockSpec(
            (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
        ),
        pl.BlockSpec(
            (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
        ),
    ]
    out_specs = [
        pl.BlockSpec(
            (None, block_seq, None, head_dim, head_dim),
            lambda i, j, k: (k, i * block_seq, j, 0, 0),
        ),
        pl.BlockSpec(
            (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
        ),
    ]

    state, normalizer = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        interpret=interpret,
        debug=debug,
        name="linear_attention_kv_state",
    )(key, value)

    # Remove padding if needed
    if pad_len > 0:
        state = state[:, :seq_len]
        normalizer = normalizer[:, :seq_len]

    return state, normalizer


def read_from_state(
    state: jax.Array,
    normalizer: jax.Array,
    query: jax.Array,
    *,
    eps: float = DEFAULT_EPS,
    block_sizes: LinearAttentionBlockSizes | None = None,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array:
    """Read from accumulated state using query.

    Args:
        state: Accumulated state, shape (B, T, H, D, D)
        normalizer: Accumulated normalizer, shape (B, T, H, D)
        query: Query with feature map applied, shape (B, T, H, D)
        eps: Epsilon for numerical stability
        block_sizes: Optional block sizes configuration
        interpret: Whether to run in interpret mode
        debug: Enable debug output

    Returns:
        Output of shape (B, T, H, D)
    """
    if block_sizes is None:
        block_sizes = LinearAttentionBlockSizes.get_default()

    batch_size, seq_len, num_heads, head_dim = query.shape
    block_seq = min(block_sizes.block_seq, seq_len)

    if seq_len % block_seq != 0:
        pad_len = block_seq - (seq_len % block_seq)
        state = jnp.pad(state, ((0, 0), (0, pad_len), (0, 0), (0, 0), (0, 0)))
        normalizer = jnp.pad(normalizer, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        query = jnp.pad(query, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        padded_seq_len = seq_len + pad_len
    else:
        padded_seq_len = seq_len
        pad_len = 0

    grid = (padded_seq_len // block_seq, num_heads, batch_size)

    out_shape = jax.ShapeDtypeStruct(
        (batch_size, padded_seq_len, num_heads, head_dim), query.dtype
    )

    kernel = functools.partial(
        _read_kernel,
        block_seq=block_seq,
        head_dim=head_dim,
        eps=eps,
    )

    in_specs = [
        pl.BlockSpec(
            (None, block_seq, None, head_dim, head_dim),
            lambda i, j, k: (k, i * block_seq, j, 0, 0),
        ),
        pl.BlockSpec(
            (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
        ),
        pl.BlockSpec(
            (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
        ),
    ]
    out_specs = pl.BlockSpec(
        (None, block_seq, None, head_dim), lambda i, j, k: (k, i * block_seq, j, 0)
    )

    output = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        debug=debug,
        name="linear_attention_read",
    )(state, normalizer, query)

    if pad_len > 0:
        output = output[:, :seq_len]

    return output


def causal_linear_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    eps: float = DEFAULT_EPS,
    interpret: bool = False,
    debug: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute causal linear attention in a single fused pass.

    This is more efficient than separate write/scan/read for causal attention
    as it maintains running state within the kernel.

    Args:
        query: Queries, shape (B, T, H, D)
        key: Keys, shape (B, T, H, D)
        value: Values, shape (B, T, H, D)
        eps: Epsilon for numerical stability
        interpret: Whether to run in interpret mode
        debug: Enable debug output

    Returns:
        Tuple of (output, final_state, final_normalizer):
            output: Shape (B, T, H, D)
            final_state: Shape (B, H, D, D)
            final_normalizer: Shape (B, H, D)
    """
    batch_size, seq_len, num_heads, head_dim = query.shape

    grid = (batch_size, num_heads)

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, seq_len, num_heads, head_dim), query.dtype),
        jax.ShapeDtypeStruct(
            (batch_size, num_heads, head_dim, head_dim), jnp.float32
        ),
        jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
    ]

    kernel = functools.partial(
        _causal_linear_attention_kernel,
        block_seq=1,  # Process one at a time for recurrence
        head_dim=head_dim,
        eps=eps,
    )

    in_specs = [
        pl.BlockSpec((None, seq_len, None, head_dim), lambda b, h: (b, 0, h, 0)),
        pl.BlockSpec((None, seq_len, None, head_dim), lambda b, h: (b, 0, h, 0)),
        pl.BlockSpec((None, seq_len, None, head_dim), lambda b, h: (b, 0, h, 0)),
    ]
    out_specs = [
        pl.BlockSpec((None, seq_len, None, head_dim), lambda b, h: (b, 0, h, 0)),
        pl.BlockSpec((None, None, head_dim, head_dim), lambda b, h: (b, h, 0, 0)),
        pl.BlockSpec((None, None, head_dim), lambda b, h: (b, h, 0)),
    ]

    output, final_state, final_normalizer = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        interpret=interpret,
        debug=debug,
        name="causal_linear_attention",
    )(query, key, value)

    return output, final_state, final_normalizer


# Pure JAX reference implementations for testing/fallback


def reference_kv_state(
    key: jax.Array, value: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Reference implementation of key-value state computation."""
    state = jnp.einsum("bthi,bthj->bthij", value, key)
    normalizer = key
    return state, normalizer


def reference_read(
    state: jax.Array,
    normalizer: jax.Array,
    query: jax.Array,
    eps: float = DEFAULT_EPS,
) -> jax.Array:
    """Reference implementation of read operation."""
    numerator = jnp.einsum("bthij,bthj->bthi", state, query)
    denominator = jnp.einsum("bthi,bthi->bth", query, normalizer)
    denominator = jnp.maximum(denominator, eps)[:, :, :, None]
    return numerator / denominator


def reference_feature_map(x: jax.Array) -> jax.Array:
    """Reference ELU+1 feature map."""
    return jnp.where(x > 0, x + 1, jnp.exp(x))
