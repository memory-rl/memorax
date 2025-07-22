import math
from typing import Tuple
import jax
from jax import numpy as jnp
import flax
import flax.linen as nn


@flax.struct.dataclass
class FFMCarry:
    memory_state: (
        jnp.ndarray
    )  # Complex-valued memory state [batch_size, memory_size, context_size]


class FFMCell(nn.RNNCellBase):
    input_size: int
    output_size: int
    memory_size: int
    context_size: int
    min_period: int = 1
    max_period: int = 1024

    @property
    def num_feature_axes(self) -> int:
        return 1

    def initialize_carry(
        self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]
    ) -> FFMCarry:
        batch_size = input_shape[0]
        memory_state = jnp.zeros(
            (batch_size, self.memory_size, self.context_size), dtype=jnp.complex64
        )
        # return FFMCarry(memory_state=memory_state)
        return memory_state

    def setup(self):
        a_low = -math.e
        a_high = -1e-6
        a = jnp.linspace(a_low, a_high, self.memory_size)
        b = (
            2
            * jnp.pi
            / jnp.linspace(self.min_period, self.max_period, self.context_size)
        )

        # Store as non-trainable parameters
        self.ffa_a = self.param("ffa_a", lambda key: a)
        self.ffa_b = self.param("ffa_b", lambda key: b)

    @nn.compact
    def __call__(
        self,
        carry: FFMCarry,
        inputs: jnp.ndarray,  # shape (batch_size, input_size)
    ) -> Tuple[jnp.ndarray, FFMCarry]:  # output shape (batch_size, output_size)
        """Process one timestep of FFM.

        Args:
            carry: Previous carry state containing memory_state and timestep
            inputs: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (output, new_carry)
            - output: shape (batch_size, output_size)
            - new_carry: Updated FFMCarry state
        """
        carry = FFMCarry(memory_state=carry)
        assert (
            inputs.ndim == 2
        ), f"Input must be 2D (batch_size, input_size), got {inputs.ndim}D"
        assert (
            inputs.shape[1] == self.input_size
        ), f"Input size must be {self.input_size}, got {inputs.shape[1]}"

        batch_size = inputs.shape[0]

        pre_linear = nn.Dense(
            self.memory_size,
            name="pre_linear",
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )

        gate_in_linear = nn.Dense(
            self.memory_size,
            name="gate_in_linear",
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )

        gate_out_linear = nn.Dense(
            self.output_size,
            name="gate_out_linear",
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )

        skip_linear = nn.Dense(
            self.output_size,
            name="skip_linear",
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )

        mix_linear = nn.Dense(
            self.output_size,
            name="mix_linear",
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )

        layer_norm = nn.LayerNorm(use_bias=False, use_scale=False, name="layer_norm")

        ffa_params = (self.ffa_a, self.ffa_b)

        gate_in = jax.nn.sigmoid(gate_in_linear(inputs))  # [batch_size, memory_size]
        gated_x = pre_linear(inputs) * gate_in  # [batch_size, memory_size]

        new_memory_state = self._apply_ffa_single_step(
            ffa_params, gated_x, carry.memory_state
        )

        # Process memory state for output
        # Concatenate real and imaginary parts: [batch_size, memory_size, context_size] -> [batch_size, 2*memory_size*context_size]
        z_in = jnp.concatenate(
            [jnp.real(new_memory_state), jnp.imag(new_memory_state)], axis=-1
        )
        z_in = z_in.reshape(batch_size, -1)  # [batch_size, 2*memory_size*context_size]

        # Mix memory representation
        z = mix_linear(z_in)  # [batch_size, output_size]

        # Output gating and skip connection
        gate_out = jax.nn.sigmoid(gate_out_linear(inputs))  # [batch_size, output_size]
        skip = skip_linear(inputs)  # [batch_size, output_size]

        out = layer_norm(z * gate_out) + skip * (1 - gate_out)

        return new_memory_state, out

    def _apply_ffa_single_step(
        self,
        ffa_params: Tuple[jnp.ndarray, jnp.ndarray],
        x: jnp.ndarray,  # [batch_size, memory_size]
        prev_state: jnp.ndarray,  # [batch_size, memory_size, context_size]
    ) -> jnp.ndarray:
        """
        Args:
            ffa_params: Tuple of (a, b) parameters for FFA
            x: Input to memory [batch_size, memory_size]
            prev_state: Previous memory state [batch_size, memory_size, context_size]
            current_timestep: Current timestep

        Returns:
            Updated memory state [batch_size, memory_size, context_size]
        """
        a, b = ffa_params

        a = jnp.clip(
            jnp.reshape(a, (1, -1, 1)), a_max=-1e-6
        )  # [memory_size] -> [1, memory_size, 1]
        b = jnp.reshape(b, (1, 1, -1))  # [context_size] -> [1, 1, context_size]

        # Compute decay factor gamma = exp(a * dt + i * b * dt)
        # For single timestep, dt = 1
        ab = jax.lax.complex(a, b)
        gamma = jnp.exp(ab * 1.0)  # [1, memory_size, context_size]

        # Expand x to match memory dimensions
        # x: [batch_size, memory_size] -> [batch_size, memory_size, context_size]
        x_expanded = jnp.expand_dims(x, axis=-1)  # [batch_size, memory_size, 1]
        x_complex = jax.lax.complex(x_expanded, jnp.zeros_like(x_expanded))
        x_complex = jnp.broadcast_to(
            x_complex, prev_state.shape
        )  # [batch_size, memory_size, context_size]

        # Update memory: new_state = gamma * prev_state + x
        # Note: For reset/done handling, this would need additional logic
        new_state = gamma * prev_state + x_complex

        return new_state


if __name__ == "__main__":
    # Example usage
    key = jax.random.PRNGKey(42)

    cell = FFMCell(input_size=64, output_size=32, memory_size=16, context_size=8)

    batch_size = 4
    input_shape = (batch_size, 64)
    carry = cell.initialize_carry(key, input_shape)

    print("Initialized carry state")
    print(f"Memory state shape: {carry.memory_state.shape}")

    inputs = jax.random.normal(key, input_shape)

    params = cell.init(key, carry, inputs)
    print("Initialized parameters")

    new_carry, output = cell.apply(params, carry, inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Memory state shape: {new_carry.memory_state.shape}")
    print("Run successful!")
