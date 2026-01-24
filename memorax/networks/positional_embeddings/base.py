from typing import Any, Optional, Protocol

from memorax.utils.typing import Array, Carry


class AbsolutePositionalEmbedding(Protocol):
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ): ...


class RelativePositionalEmbedding(Protocol):
    def __call__(
        self, query: Array, key: Array, query_pos: Array, key_pos: Array
    ) -> tuple[Array, Array, Any]: ...
