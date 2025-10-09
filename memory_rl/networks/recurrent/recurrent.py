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

    def _resolve_runtime_options(
        self,
        return_carry,
        return_carry_history,
        time_major,
        reverse,
        keep_order,
        burn_in_length,
        detach_after_burn_in,
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
        return (
            return_carry,
            return_carry_history,
            time_major,
            reverse,
            keep_order,
            burn_in_length,
            detach_after_burn_in,
        )

    def _infer_time_axis_and_batch_dims(self, inputs: jax.Array, time_major: bool):
        time_axis = 0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)
        if time_axis < 0:
            time_axis += inputs.ndim
        if time_major:
            batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
        else:
            batch_dims = inputs.shape[:time_axis]
        return time_axis, batch_dims

    def _maybe_reverse(
        self,
        inputs,
        mask,
        *,
        seq_lengths,
        batch_dims,
        time_major,
        reverse,
    ):
        if not reverse:
            return inputs, mask
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
        return inputs, mask

    def _initialize_carry(self, initial_carry, init_key, inputs, time_axis):
        if initial_carry is not None:
            return initial_carry
        if init_key is None:
            init_key = jax.random.key(0)
        input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
        return self.cell.initialize_carry(init_key, input_shape)

    def _build_scan(
        self, slice_carry: bool, return_carry_history: bool, inputs, time_axis: int
    ):
        input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
        initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)
        def scan_fn(cell, carry, x, mask):
            carry = jax.tree.map(
                lambda initial_carry, carry: jnp.where(
                    self._broadcast(mask, carry),
                    initial_carry,
                    carry,
                ),
                initial_carry,
                carry,
            )
            carry, y = cell(carry, x)
            if slice_carry or return_carry_history:
                return carry, (carry, y)
            else:
                return carry, y

        return nn.scan(
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

    @staticmethod
    def _slice_time(x, low, high, time_axis: int):
        time_slice = [slice(None)] * x.ndim
        time_slice[time_axis] = slice(low, high)
        return x[tuple(time_slice)]

    @staticmethod
    def _concat_time(a, b, axis: int):
        if a is None:
            return b
        if b is None:
            return a
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=axis), a, b)

    def _restore_order_if_needed(
        self, outputs, *, reverse, keep_order, seq_lengths, batch_dims, time_major
    ):
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
        return outputs

    def burn_in(
        self,
        *,
        scan,
        carry,
        inputs,
        mask,
        seq_lengths,
        time_axis,
        B,
        slice_carry,
        return_carry_history,
    ):
        carries_burn = outputs_burn = None
        if B > 0:
            burn_out = scan(
                self.cell,
                carry,
                self._slice_time(inputs, 0, B, time_axis),
                self._slice_time(mask, 0, B, time_axis),
            )
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
        return carries_burn, outputs_burn, carry_after_burn

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
        (
            return_carry,
            return_carry_history,
            time_major,
            reverse,
            keep_order,
            burn_in_length,
            detach_after_burn_in,
        ) = self._resolve_runtime_options(
            return_carry,
            return_carry_history,
            time_major,
            reverse,
            keep_order,
            burn_in_length,
            detach_after_burn_in,
        )

        time_axis, batch_dims = self._infer_time_axis_and_batch_dims(inputs, time_major)

        inputs, mask = self._maybe_reverse(
            inputs,
            mask,
            seq_lengths=seq_lengths,
            batch_dims=batch_dims,
            time_major=time_major,
            reverse=reverse,
        )

        carry: Any = self._initialize_carry(initial_carry, init_key, inputs, time_axis)

        slice_carry = seq_lengths is not None and return_carry

        scan = self._build_scan(slice_carry, return_carry_history, inputs, time_axis)

        has_learn = (int(inputs.shape[time_axis]) - int(burn_in_length)) > 0

        carries_burn, outputs_burn, carry_after_burn = self.burn_in(
            scan=scan,
            carry=carry,
            inputs=inputs,
            mask=mask,
            seq_lengths=seq_lengths,
            time_axis=time_axis,
            B=burn_in_length,
            slice_carry=slice_carry,
            return_carry_history=return_carry_history,
        )

        if detach_after_burn_in:
            carry_learn = jax.tree.map(jax.lax.stop_gradient, carry_after_burn)
        else:
            carry_learn = carry_after_burn

        carries_learn = outputs_learn = None
        if has_learn:
            use_parallel = hasattr(self.cell, "apply_parallel") and inputs.shape[time_axis] > 1 and not (
                slice_carry or return_carry_history
            )
            if use_parallel:

                inputs_learn = self._slice_time(inputs, burn_in_length, None, time_axis)
                mask_learn = self._slice_time(mask, burn_in_length, None, time_axis)
                carries_learn, outputs_learn = self.cell.apply_parallel(
                    carry_learn,
                    inputs_learn,
                    mask_learn,
                )
                carry_learn_final = carries_learn
            else:
                learn_out = scan(
                    self.cell,
                    carry_learn,
                    self._slice_time(inputs, burn_in_length, None, time_axis),
                    self._slice_time(mask, burn_in_length, None, time_axis),
                )
                if slice_carry or return_carry_history:
                    _, (carries_learn, outputs_learn) = learn_out
                else:
                    carry_learn_final, outputs_learn = learn_out
        else:
            carry_learn_final = carry_learn

        if slice_carry:
            assert seq_lengths is not None
            carries_full = self._concat_time(carries_burn, carries_learn, axis=0)
            outputs = self._concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = nn._select_last_carry(carries_full, seq_lengths)
        elif return_carry_history:
            carries_full = self._concat_time(carries_burn, carries_learn, axis=0)
            outputs = self._concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = carries_full
        else:
            outputs = self._concat_time(outputs_burn, outputs_learn, axis=time_axis)
            carry_final = carry_learn_final

        outputs = self._restore_order_if_needed(
            outputs,
            reverse=reverse,
            keep_order=keep_order,
            seq_lengths=seq_lengths,
            batch_dims=batch_dims,
            time_major=time_major,
        )

        if return_carry or return_carry_history:
            return carry_final, outputs
        else:
            return outputs
