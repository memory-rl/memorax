import flax.linen as nn
import jax
import jax.numpy as jnp


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
            inputs = jax.tree_util.tree_map(
                lambda x: nn.flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                inputs,
            )
            mask = jax.tree_util.tree_map(
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
            outputs = jax.tree_util.tree_map(
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
