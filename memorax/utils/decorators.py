import functools

import jax


def callback(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        jax.debug.callback(lambda args, kwargs: function(*args, **kwargs), args, kwargs)

    return wrapper
