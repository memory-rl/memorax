from typing import Any, Protocol

from memorax.utils.typing import Array


class PositionalEmbedding(Protocol):
    def __call__(
        self, query: Array, key: Array, query_pos: Array, key_pos: Array
    ) -> tuple[Array, Array, Any]: ...
