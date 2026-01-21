import functools

import jax


def callback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return jax.debug.callback(f, *args, **kwargs)

    return wrapper
