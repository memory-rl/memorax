from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import Dense, default_kernel_init
from flax.typing import Array, Dtype, Initializer, PRNGKey
from jax import random


class mLSTMCell(nn.RNNCellBase):
    """Multiplicative LSTM (mLSTM) cell as described in *xLSTM: Scaling RNNs*."""

    features: int
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @nn.compact
    def __call__(self, carry: Tuple[Array, Array], inputs: Array):
        c_prev, h_prev = carry
        hidden_size = h_prev.shape[-1]

        dense_xm = Dense(
            features=hidden_size,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="xm",
        )
        dense_hm = Dense(
            features=hidden_size,
            use_bias=False,
            kernel_init=self.recurrent_kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="hm",
        )
        m_t = dense_xm(inputs) * dense_hm(h_prev)

        def dense_x(name):
            return Dense(
                features=hidden_size,
                use_bias=False,
                kernel_init=self.kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"x{name}",
            )

        def dense_m(name):
            return Dense(
                features=hidden_size,
                use_bias=True,  # bias lives in the multiplicative path
                kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"m{name}",
            )

        i = self.gate_fn(dense_x("i")(inputs) + dense_m("i")(m_t))
        f = self.gate_fn(dense_x("f")(inputs) + dense_m("f")(m_t))
        g = self.activation_fn(dense_x("g")(inputs) + dense_m("g")(m_t))
        o = self.gate_fn(dense_x("o")(inputs) + dense_m("o")(m_t))

        c = f * c_prev + i * g
        h = o * self.activation_fn(c)
        return (c, h), h

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize cell + hidden state to zeros."""
        batch_shape = input_shape[:-1]
        key_c, key_h = random.split(rng)
        state_shape = batch_shape + (self.features,)
        c0 = self.carry_init(key_c, state_shape, self.param_dtype)
        h0 = self.carry_init(key_h, state_shape, self.param_dtype)
        return (c0, h0)

    @property
    def num_feature_axes(self) -> int:
        return 1


class MaskedmLSTMCell(mLSTMCell):

    @nn.compact
    def __call__(self, carry, inputs, mask):
        c, h = carry
        initial_c, initial_h = self.initialize_carry(jax.random.key(0), inputs.shape)
        c = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_c,
            c,
        )
        h = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_h,
            h,
        )
        return super().__call__((c, h), inputs)


class MaskedGRUCell(nn.GRUCell):

    @nn.compact
    def __call__(self, carry, inputs, mask):
        h = carry
        h = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            self.initialize_carry(jax.random.key(0), inputs.shape),
            h,
        )

        return super().__call__(h, inputs)


class MaskedLSTMCell(nn.LSTMCell):

    @nn.compact
    def __call__(self, carry, inputs, mask):
        c, h = carry
        initial_c, initial_h = self.initialize_carry(jax.random.key(0), inputs.shape)
        c = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_c,
            c,
        )
        h = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_h,
            h,
        )

        return super().__call__((c, h), inputs)


class MaskedOptimizedLSTMCell(nn.OptimizedLSTMCell):

    @nn.compact
    def __call__(self, carry, inputs, mask):
        c, h = carry
        initial_c, initial_h = self.initialize_carry(jax.random.key(0), inputs.shape)
        c = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_c,
            c,
        )
        h = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            initial_h,
            h,
        )

        return super().__call__((c, h), inputs)


class MaskedRNN(nn.RNN):

    return_carry_history: bool = False
    burn_in_length: int = 0

    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        *,
        initial_carry=None,
        init_key=None,
        seq_lengths=None,
        return_carry=None,
        return_carry_history=None,
        time_major=None,
        reverse=None,
        keep_order=None,
        burn_in_length=None,
    ):

        if return_carry is None:
            return_carry = self.return_carry
        if return_carry_history is None:
            return_carry_history = self.return_carry_history
        if time_major is None:
            time_major = self.time_major
        if reverse is None:
            reverse = self.reverse
        if keep_order is None:
            keep_order = self.keep_order
        if burn_in_length is None:
            burn_in_length = self.burn_in_length

        # Infer the number of batch dimensions from the input shape.
        # Cells like ConvLSTM have additional spatial dimensions.
        time_axis = 0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)

        # make time_axis positive
        if time_axis < 0:
            time_axis += inputs.ndim

        if time_major:
            # we add +1 because we moved the time axis to the front
            batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
        else:
            batch_dims = inputs.shape[:time_axis]

        # maybe reverse the sequence
        if reverse:
            inputs = jax.tree.map(
                lambda x: nn.flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                inputs,
            )
            mask = jax.tree.map(
                lambda m: nn.flip_sequences(
                    m,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                mask,
            )

        carry: Any
        if initial_carry is None:
            if init_key is None:
                init_key = jax.random.key(0)

            input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
            carry = self.cell.initialize_carry(init_key, input_shape)
        else:
            carry = initial_carry

        slice_carry = seq_lengths is not None and return_carry

        def scan_fn(cell, carry, x, mask):
            carry, y = cell(carry, x, mask)
            # When we have a segmentation mask we return the carry as an output
            # so that we can select the last carry for each sequence later.
            # This uses more memory but is faster than using jnp.where at each
            # iteration. As a small optimization do this when we really need it.
            if slice_carry or return_carry_history:
                return carry, (carry, y)
            else:
                return carry, y

        scan = nn.transforms.scan(
            scan_fn,
            in_axes=(time_axis, time_axis),
            out_axes=(
                (0, time_axis) if (slice_carry or return_carry_history) else time_axis
            ),
            unroll=self.unroll,
            variable_axes=self.variable_axes,
            variable_broadcast=self.variable_broadcast,
            variable_carry=self.variable_carry,
            split_rngs=self.split_rngs,
        )

        if burn_in_length > 0:
            burn_in_inputs, inputs = jnp.split(inputs, [burn_in_length], axis=time_axis)
            burn_in_mask, mask = jnp.split(mask, [burn_in_length], axis=time_axis)

            scan_output = scan(self.cell, carry, burn_in_inputs, burn_in_mask)
            if slice_carry or return_carry_history:
                final_burn_in_carry, (burn_in_carry, outputs) = scan_output
            else:
                final_burn_in_carry, outputs = scan_output
            carry = jax.tree.map(jax.lax.stop_gradient, final_burn_in_carry)

        scan_output = scan(self.cell, carry, inputs, mask)

        # Next we select the final carry. If a segmentation mask was provided and
        # return_carry is True we slice the carry history and select the last valid
        # carry for each sequence. Otherwise we just use the last carry.
        if slice_carry:
            assert seq_lengths is not None
            _, (carries, outputs) = scan_output
            # seq_lengths[None] expands the shape of the mask to match the
            # number of dimensions of the carry.
            carry = nn._select_last_carry(carries, seq_lengths)
        elif return_carry_history:
            _, (carry, outputs) = scan_output
        else:
            carry, outputs = scan_output

        if reverse and keep_order:
            outputs = jax.tree.map(
                lambda x: nn.flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                outputs,
            )

        if return_carry or return_carry_history:
            return carry, outputs
        else:
            return outputs
