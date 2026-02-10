import functools
import time
from typing import Any, Callable

import jax


def profile(fn: Callable, num_steps, exclude_jit=True):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        if exclude_jit:
            _ = fn(*args, **kwargs)
            jax.block_until_ready(_)

        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        result = jax.block_until_ready(result)
        end_time = time.perf_counter()

        run_time = num_steps / (end_time - start_time)
        return result, run_time

    return wrapper
