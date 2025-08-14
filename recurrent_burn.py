from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class MaskedRNN(nn.RNN):

    return_carry_history: bool = False
    burn_in_length: int = 0
    detach_after_burn_in: bool = False

    @staticmethod
    def _broadcast(mask: jax.Array, carry: jax.Array) -> jax.Array:
        while mask.ndim < carry.ndim:
            mask = mask[..., None]
        return mask

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
        detach_after_burn_in=None,
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
        if detach_after_burn_in is None:
            detach_after_burn_in = self.detach_after_burn_in

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
            carry = jax.tree.map(
                lambda initial_carry, carry: jnp.where(
                    self._broadcast(mask, carry),
                    initial_carry,
                    carry,
                ),
                self.cell.initialize_carry(jax.random.key(0), x.shape),
                carry,
            )
            carry, y = cell(carry, x)
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

        def _slice_time(x, low, high):
            time_slice = [slice(None)] * x.ndim
            time_slice[time_axis] = slice(low, high)
            return x[time_slice]

        T = inputs.shape[time_axis]
        B = int(jnp.clip(burn_in_length, 0, T))
        has_learn = (T - B) > 0

        carries_burn = outputs_burn = None
        if B > 0:
            burn_out = scan(self.cell, carry, _slice_time(inputs, 0, B), _slice_time(mask, 0, B))
            if slice_carry:
                _, (carries_burn, outputs_burn) = burn_out
                burn_lengths = jnp.minimum(seq_lengths, B)
                carry_after_burn = nn._select_last_carry(carries_burn, burn_lengths)
            elif return_carry_history:
                _, (carries_burn, outputs_burn) = burn_out
                carry_after_burn = jax.tree.map(lambda c: c[-1], carries_burn)
            else:
                carry_after_burn, outputs_burn = burn_out
        else:
            carry_after_burn = carry

        if detach_after_burn_in:
            carry_learn = jax.tree.map(jax.lax.stop_gradient, carry_after_burn)
        else:
            carry_learn = carry_after_burn

        carries_learn = outputs_learn = None
        if has_learn:
            learn_out = scan(self.cell, carry_learn, _slice_time(inputs, B, None), _slice_time(mask, B, None))
            if slice_carry or return_carry_history:
                _, (carries_learn, outputs_learn) = learn_out
            else:
                carry_learn_final, outputs_learn = learn_out
        else:
            carry_learn_final = carry_learn

        def _concat_time(a, b, axis):
            if a is None: return b
            if b is None: return a
            return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=axis), a, b)

        # Next we select the final carry. If a segmentation mask was provided and
        # return_carry is True we slice the carry history and select the last valid
        # carry for each sequence. Otherwise we just use the last carry.
        if slice_carry:
            assert seq_lengths is not None
            carries_full = _concat_time(carries_burn, carries_learn, axis=0)  # time-first for carries
            outputs = _concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = nn._select_last_carry(carries_full, seq_lengths)
        elif return_carry_history:
            carries_full = _concat_time(carries_burn, carries_learn, axis=0)
            outputs = _concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = carries_full
        else:
            outputs = _concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = carry_learn_final

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
            return carry_final, outputs
        else:
            return outputs

