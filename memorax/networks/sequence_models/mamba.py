from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.networks.sequence_models.utils import (add_time_axis,
                                                    get_input_shape)

from .sequence_model import SequenceModel


@jax.vmap
def _binary_operator(q_i, q_j):
    a_i, b_i, c_i = q_i
    a_j, b_j, c_j = q_j

    a_out = (a_j * a_i) * (1.0 - c_j) + a_j * c_j
    b_out = (a_j * b_i + b_j) * (1.0 - c_j) + b_j * c_j
    c_out = c_i * (1.0 - c_j) + c_j
    return a_out, b_out, c_out


def a_log_init(key, shape, dtype=jnp.float32):
    return jax.random.uniform(key, shape, minval=-3.0, maxval=3.0, dtype=dtype)


class MambaLayer(nn.Module):
    features: int
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, dt, B, C, mask, initial_carry):
        log_decay = self.param(
            "log_decay", nn.initializers.normal(stddev=0.1), (self.num_heads,)
        )
        skip_weight = self.param(
            "skip_weight", nn.initializers.ones, (self.num_heads, self.head_dim)
        )

        decay_rate = -jnp.exp(log_decay)
        decay_factor = jnp.exp(dt * decay_rate[None, :])

        decay_factor = decay_factor[:, :, None, None]
        decay_factor = jnp.concatenate(
            [jnp.ones((1, self.num_heads, 1, 1)), decay_factor], axis=0
        )

        driving_input = jnp.einsum("thn,thd->thnd", B * dt[:, :, None], x)

        driving_input = jnp.concatenate(
            [initial_carry[None, ...], driving_input], axis=0
        )

        mask = jnp.concatenate([jnp.zeros(1), mask])
        mask = add_time_axis(mask)

        _, hidden_states, _ = jax.lax.associative_scan(
            _binary_operator, (decay_factor, driving_input, mask)
        )

        next_carry = hidden_states[-1]
        valid_states = hidden_states[1:]

        output = jnp.einsum("thn,thnd->thd", C, valid_states)

        output = output + skip_weight[None, :, :] * x

        return next_carry, output


class MambaBlock(nn.Module):
    features: int
    expansion_factor: int = 2
    hidden_dim: int = 64
    conv_kernel_size: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    training: bool = True
    prenorm: bool = False

    @nn.compact
    def __call__(self, inputs, mask, carry):
        expanded_dim = self.features * self.expansion_factor
        head_dim = expanded_dim // self.num_heads

        dim_gate = expanded_dim
        dim_signal = expanded_dim
        dim_dynamics = self.num_heads * self.hidden_dim
        dim_readout = self.num_heads * self.hidden_dim
        dim_dt = self.num_heads

        # The convolution branch includes Signal, B, and C
        dim_conv_block = dim_signal + dim_dynamics + dim_readout

        # --- 2. Pre-Norm ---
        x = inputs
        if self.prenorm:
            x = nn.LayerNorm(name="pre_norm")(x)

        # --- 3. Fused Projection ---
        # Projects input to: [Gate Branch, Conv Branch (Signal+B+C), TimeStep]
        total_proj_dim = dim_gate + dim_conv_block + dim_dt

        projected_mixed = nn.Dense(total_proj_dim, name="in_proj")(x)

        # Split 1: Separate the Gate, the Conv Input, and the Time Step
        # Indices: [gate_end, conv_end]
        split_points = [dim_gate, dim_gate + dim_conv_block]
        gate_branch, conv_branch, raw_dt = jnp.split(
            projected_mixed, split_points, axis=-1
        )

        # --- 4. Local Mixing (Depthwise Convolution) ---
        # This gives the model local context before the global SSM scan
        if self.conv_kernel_size > 1:
            # Causal padding: (kernel - 1) on the left
            conv_branch = jnp.pad(
                conv_branch, ((0, 0), (self.conv_kernel_size - 1, 0), (0, 0))
            )

            conv_branch = nn.Conv(
                features=dim_conv_block,  # Output features same as input
                kernel_size=(self.conv_kernel_size,),
                feature_group_count=dim_conv_block,  # Depthwise convolution!
                padding="VALID",
                name="conv1d",
            )(conv_branch)

        conv_branch = nn.silu(conv_branch)

        # Split 2: Unpack the Conv Branch into specific SSM components
        # Indices: [signal_end, dynamics_end]
        split_points_conv = [dim_signal, dim_signal + dim_dynamics]
        ssm_signal, ssm_dynamics, ssm_readout = jnp.split(
            conv_branch, split_points_conv, axis=-1
        )

        # --- 5. Reshape for Multi-Head SSM ---
        batch, length, _ = ssm_signal.shape

        # Reshape Signal (x) -> [Batch, Length, Heads, Head_Dim]
        ssm_signal = ssm_signal.reshape(batch, length, self.num_heads, head_dim)

        # Reshape Parameters (dt, B, C) -> [Batch, Length, Heads, State_Dim or 1]
        dt_softplus = nn.softplus(raw_dt).reshape(batch, length, self.num_heads)
        ssm_dynamics = ssm_dynamics.reshape(
            batch, length, self.num_heads, self.hidden_dim
        )
        ssm_readout = ssm_readout.reshape(
            batch, length, self.num_heads, self.hidden_dim
        )

        carry, ssm_output = nn.vmap(
            MambaLayer,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(
            features=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=head_dim,
            name="ssm",
        )(ssm_signal, dt_softplus, ssm_dynamics, ssm_readout, mask, carry)

        # Flatten heads back to [Batch, Length, Expanded_Dim]
        ssm_output = ssm_output.reshape(batch, length, expanded_dim)

        # --- 7. Post-SSM Norm & Gating ---
        # GroupNorm ensures stability across the independent heads
        ssm_output = nn.GroupNorm(num_groups=self.num_heads, name="ssm_norm")(
            ssm_output
        )

        # Multiplicative Gating (GLU-style)
        # The gate branch decides what information flows through
        gated_output = ssm_output * nn.silu(gate_branch)

        # --- 8. Output Projection ---
        output = nn.Dense(self.features, name="out_proj")(gated_output)

        output = nn.Dropout(
            self.dropout_rate, broadcast_dims=(0,), deterministic=not self.training
        )(output)

        # Residual connection
        output = inputs + output

        if not self.prenorm:
            output = nn.LayerNorm(name="post_norm")(output)

        return carry, output


class Mamba(SequenceModel):
    features: int
    num_layers: int
    expansion_factor: int = 2
    hidden_dim: int = 64
    num_heads: int = 8
    dropout_rate: float = 0.0
    training: bool = True
    prenorm: bool = False

    @nn.compact
    def __call__(self, inputs, mask, initial_carry=None, **kwargs):
        x = inputs
        new_carries = []

        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        for i, carry_i in enumerate(initial_carry):
            new_carry_i, x = MambaBlock(
                features=self.features,
                expansion_factor=self.expansion_factor,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                training=self.training,
                prenorm=self.prenorm,
                name=f"layer_{i}",
            )(x, mask, carry_i)
            new_carries.append(new_carry_i)

        return tuple(new_carries), x

    def initialize_carry(self, key: jax.Array, input_shape) -> tuple:
        batch_size, *_ = input_shape
        inner_dim = self.features * self.expansion_factor
        head_dim = inner_dim // self.num_heads

        initial_carry = jnp.zeros(
            (batch_size, self.num_heads, self.hidden_dim, head_dim),
            dtype=jnp.float32,
        )
        return tuple(initial_carry for _ in range(self.num_layers))
