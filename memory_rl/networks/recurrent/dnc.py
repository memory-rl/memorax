from typing import Any, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.recurrent import RNNCellBase
from flax import struct
from flax.typing import Key


@struct.dataclass
class TemporalLinkageState:
    link: jnp.ndarray
    precedence_weights: jnp.ndarray


@struct.dataclass
class AccessState:
    memory: jnp.ndarray
    read_weights: jnp.ndarray
    write_weights: jnp.ndarray
    linkage: TemporalLinkageState
    usage: jnp.ndarray


@struct.dataclass
class DNCState:
    access_output: jnp.ndarray
    access_state: AccessState
    controller_state: Any


class CosineWeights(nn.Module):
    num_heads: int
    word_size: int
    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, memory: jnp.ndarray, keys: jnp.ndarray, strengths: jnp.ndarray
    ) -> jnp.ndarray:
        memory = memory.astype(self.dtype)
        keys = keys.astype(self.dtype)
        strengths = strengths.astype(self.dtype)
        dot = jnp.einsum("bhw,bnw->bhn", keys, memory)
        mem_norms = jnp.linalg.norm(memory, axis=2, keepdims=True)
        key_norms = jnp.linalg.norm(keys, axis=2, keepdims=True)
        denom = key_norms * jnp.swapaxes(mem_norms, 1, 2)
        sim = dot / (denom + self.epsilon)
        sharp = sim * jax.nn.softplus(strengths)[..., None]
        return nn.softmax(sharp, axis=-1)


class TemporalLinkage(nn.Module):
    memory_size: int
    num_writes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, write_weights: jnp.ndarray, prev_state: TemporalLinkageState
    ) -> TemporalLinkageState:
        write_weights = write_weights.astype(self.dtype)
        link = prev_state.link.astype(self.dtype)
        precedence_weights = prev_state.precedence_weights.astype(self.dtype)
        n = self.memory_size
        ww_i = write_weights[:, :, :, None]
        ww_j = write_weights[:, :, None, :]
        pw_j = precedence_weights[:, :, None, :]
        prev_link_scale = 1.0 - ww_i - ww_j
        new_link = ww_i * pw_j
        link = prev_link_scale * link + new_link
        eye = jnp.eye(n, dtype=link.dtype)
        link = link * (1.0 - eye[None, None, :, :])
        write_sum = jnp.sum(write_weights, axis=-1, keepdims=True)
        precedence = (1.0 - write_sum) * precedence_weights + write_weights
        return TemporalLinkageState(link=link, precedence_weights=precedence)

    def directional_read_weights(
        self, link: jnp.ndarray, prev_read_weights: jnp.ndarray, forward: bool
    ) -> jnp.ndarray:
        exp_rw = jnp.repeat(prev_read_weights[:, None, :, :], link.shape[1], axis=1)
        lnk = jnp.swapaxes(link, -2, -1) if forward else link
        result = jnp.einsum("bwri,bwij->bwrj", exp_rw, lnk)
        return jnp.transpose(result, (0, 2, 1, 3))



class Freeness(nn.Module):
    memory_size: int
    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        write_weights: jnp.ndarray,
        free_gate: jnp.ndarray,
        read_weights: jnp.ndarray,
        prev_usage: jnp.ndarray,
    ) -> jnp.ndarray:
        write_weights = jax.lax.stop_gradient(write_weights).astype(self.dtype)
        usage = self._usage_after_write(prev_usage.astype(self.dtype), write_weights)
        usage = self._usage_after_read(
            usage, free_gate.astype(self.dtype), read_weights.astype(self.dtype)
        )
        return usage

    def write_allocation_weights(
        self, usage: jnp.ndarray, write_gates: jnp.ndarray, num_writes: int
    ) -> jnp.ndarray:
        write_gates = write_gates[..., None]
        allocs = []
        u = usage
        for i in range(num_writes):
            a = self._allocation(u)
            allocs.append(a)
            u = u + (1.0 - u) * write_gates[:, i, :] * a
        return jnp.stack(allocs, axis=1)

    def _usage_after_write(
        self, prev_usage: jnp.ndarray, write_weights: jnp.ndarray
    ) -> jnp.ndarray:
        w = 1.0 - jnp.prod(1.0 - write_weights, axis=1)
        return prev_usage + (1.0 - prev_usage) * w

    def _usage_after_read(
        self, prev_usage: jnp.ndarray, free_gate: jnp.ndarray, read_weights: jnp.ndarray
    ) -> jnp.ndarray:
        free_gate = free_gate[..., None]
        free_read_weights = free_gate * read_weights
        phi = jnp.prod(1.0 - free_read_weights, axis=1)
        return prev_usage * phi

    def _allocation(self, usage: jnp.ndarray) -> jnp.ndarray:
        usage = self.epsilon + (1.0 - self.epsilon) * usage
        nonusage = 1.0 - usage
        indices = jnp.argsort(-nonusage, axis=1)
        sorted_nonusage = jnp.take_along_axis(nonusage, indices, axis=1)
        sorted_usage = 1.0 - sorted_nonusage
        prod_sorted_usage = jnp.cumprod(sorted_usage, axis=1)
        prod_exclusive = jnp.concatenate(
            [jnp.ones_like(prod_sorted_usage[:, :1]), prod_sorted_usage[:, :-1]], axis=1
        )
        sorted_alloc = sorted_nonusage * prod_exclusive
        b, n = indices.shape
        inv = jnp.zeros_like(indices)
        inv = inv.at[jnp.arange(b)[:, None], indices].set(jnp.arange(n)[None, :])
        return jnp.take_along_axis(sorted_alloc, inv, axis=1)



def _erase_and_write(
    memory: jnp.ndarray,
    address: jnp.ndarray,
    reset_weights: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    expand_address_i = address[:, :, :, None]
    reset_weights_j = reset_weights[:, :, None, :]
    weighted_resets = expand_address_i * reset_weights_j
    reset_gate = jnp.prod(1.0 - weighted_resets, axis=1)
    memory = memory * reset_gate
    add_matrix = jnp.einsum("bmn,bmw->bnw", address, values)
    return memory + add_matrix


class MemoryAccess(nn.Module):
    memory_size: int = 128
    word_size: int = 20
    num_reads: int = 1
    num_writes: int = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, prev_state: AccessState
    ) -> Tuple[jnp.ndarray, AccessState]:
        x = inputs.astype(self.dtype)
        num_read_modes = 1 + 2 * self.num_writes

        def linear_stack(
            first_dim: int, second_dim: int, name: str, act: Optional[Any] = None
        ) -> jnp.ndarray:
            y = nn.Dense(
                first_dim * second_dim,
                name=name,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
            if act is not None:
                y = act(y)
            b = y.shape[0]
            return y.reshape((b, first_dim, second_dim))

        write_vectors = linear_stack(self.num_writes, self.word_size, "write_vectors")
        erase_vectors = linear_stack(
            self.num_writes, self.word_size, "erase_vectors", jax.nn.sigmoid
        )
        free_gate = jax.nn.sigmoid(
            nn.Dense(
                self.num_reads,
                name="free_gate",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
        )
        allocation_gate = jax.nn.sigmoid(
            nn.Dense(
                self.num_writes,
                name="allocation_gate",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
        )
        write_gate = jax.nn.sigmoid(
            nn.Dense(
                self.num_writes,
                name="write_gate",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
        )
        read_mode_logits = linear_stack(self.num_reads, num_read_modes, "read_mode")
        read_mode = nn.softmax(read_mode_logits, axis=-1)
        write_keys = linear_stack(self.num_writes, self.word_size, "write_keys")
        write_strengths = nn.Dense(
            self.num_writes,
            name="write_strengths",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        read_keys = linear_stack(self.num_reads, self.word_size, "read_keys")
        read_strengths = nn.Dense(
            self.num_reads,
            name="read_strengths",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

        usage = Freeness(self.memory_size, dtype=self.dtype)(
            prev_state.write_weights,
            free_gate,
            prev_state.read_weights,
            prev_state.usage,
        )
        write_content_weights = CosineWeights(
            self.num_writes, self.word_size, dtype=self.dtype
        )(prev_state.memory, write_keys, write_strengths)
        write_allocation_weights = Freeness(
            self.memory_size, dtype=self.dtype
        ).write_allocation_weights(usage, allocation_gate * write_gate, self.num_writes)
        ag = allocation_gate[..., None]
        wg = write_gate[..., None]
        write_weights = wg * (
            ag * write_allocation_weights + (1.0 - ag) * write_content_weights
        )
        memory = _erase_and_write(
            prev_state.memory, write_weights, erase_vectors, write_vectors
        )
        linkage_state = TemporalLinkage(
            self.memory_size, self.num_writes, dtype=self.dtype
        )(write_weights, prev_state.linkage)
        content_weights = CosineWeights(
            self.num_reads, self.word_size, dtype=self.dtype
        )(memory, read_keys, read_strengths)
        forward_weights = TemporalLinkage(
            self.memory_size, self.num_writes, dtype=self.dtype
        ).directional_read_weights(linkage_state.link, prev_state.read_weights, True)
        backward_weights = TemporalLinkage(
            self.memory_size, self.num_writes, dtype=self.dtype
        ).directional_read_weights(linkage_state.link, prev_state.read_weights, False)
        backward_mode = read_mode[:, :, : self.num_writes]
        forward_mode = read_mode[:, :, self.num_writes : 2 * self.num_writes]
        content_mode = read_mode[:, :, 2 * self.num_writes]
        read_weights = (
            content_mode[..., None] * content_weights
            + jnp.sum(forward_mode[..., None] * forward_weights, axis=2)
            + jnp.sum(backward_mode[..., None] * backward_weights, axis=2)
        )
        read_words = jnp.einsum("brn,bnw->brw", read_weights, memory)
        next_state = AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage,
        )
        return read_words, next_state

    @property
    def output_size(self) -> Tuple[int, int]:
        return self.num_reads, self.word_size



class DNCCell(RNNCellBase):
    features: int
    controller_features: int = 256
    memory_size: int = 128
    word_size: int = 20
    num_reads: int = 1
    num_writes: int = 1
    clip_value: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = None
    bias_init: Any = None

    @nn.compact
    def __call__(
        self, carry: DNCState, inputs: jnp.ndarray
    ) -> Tuple[DNCState, jnp.ndarray]:
        prev_access_output = carry.access_output
        prev_access_state = carry.access_state
        prev_controller_state = carry.controller_state
        x = inputs.reshape((inputs.shape[0], -1)).astype(self.dtype)
        prev_read_flat = prev_access_output.reshape(
            (prev_access_output.shape[0], -1)
        ).astype(self.dtype)
        controller_in = jnp.concatenate([x, prev_read_flat], axis=-1)
        controller = nn.LSTMCell(
            features=self.controller_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            # name="controller",
        )
        controller_state, controller_output = controller(
            prev_controller_state, controller_in
        )
        if self.clip_value > 0.0:
            controller_output = jnp.clip(
                controller_output, -self.clip_value, self.clip_value
            )
            controller_state = controller_state.replace(
                carry=jnp.clip(
                    controller_state.carry, -self.clip_value, self.clip_value
                ),
                hidden=jnp.clip(
                    controller_state.hidden, -self.clip_value, self.clip_value
                ),
            )
        access = MemoryAccess(
            self.memory_size,
            self.word_size,
            self.num_reads,
            self.num_writes,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="access",
        )
        access_output, access_state = access(controller_output, prev_access_state)
        out_in = jnp.concatenate(
            [controller_output, access_output.reshape((access_output.shape[0], -1))],
            axis=-1,
        )
        y = nn.Dense(
            self.features,
            name="output_linear",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(out_in)
        if self.clip_value > 0.0:
            y = jnp.clip(y, -self.clip_value, self.clip_value)
        new_carry = DNCState(
            access_output=access_output,
            access_state=access_state,
            controller_state=controller_state,
        )
        return new_carry, y

    def initialize_carry(self, rng: Key, input_shape: Tuple[int, ...]) -> DNCState:
        batch_size = input_shape[0]
        memory = jnp.zeros((batch_size, self.memory_size, self.word_size), self.dtype)
        read_weights = jnp.zeros(
            (batch_size, self.num_reads, self.memory_size), self.dtype
        )
        write_weights = jnp.zeros(
            (batch_size, self.num_writes, self.memory_size), self.dtype
        )
        link = jnp.zeros(
            (batch_size, self.num_writes, self.memory_size, self.memory_size),
            self.dtype,
        )
        precedence = jnp.zeros(
            (batch_size, self.num_writes, self.memory_size), self.dtype
        )
        usage = jnp.zeros((batch_size, self.memory_size), self.dtype)
        access_state = AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=TemporalLinkageState(link=link, precedence_weights=precedence),
            usage=usage,
        )
        access_output = jnp.zeros(
            (batch_size, self.num_reads, self.word_size), self.dtype
        )

        c = jnp.zeros((batch_size, self.controller_features), self.param_dtype)
        h = jnp.zeros((batch_size, self.controller_features), self.param_dtype)

        return DNCState(
            access_output=access_output,
            access_state=access_state,
            controller_state=(c, h),
        )

    @property
    def num_feature_axes(self) -> int:
        return 1
