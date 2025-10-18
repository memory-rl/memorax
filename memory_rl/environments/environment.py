from memory_rl.environments import (
    brax,
    craftax,
    gxm,
    gymnax,
    mujoco,
    navix,
    popgym_arcade,
    popjym,
)

register = {
    "brax": brax.make,
    "craftax": craftax.make,
    "gymnax": gymnax.make,
    "gxm": gxm.make,
    "mujoco": mujoco.make,
    "navix": navix.make,
    "popgym_arcade": popgym_arcade.make,
    "popjym": popjym.make,
}


def make(
    env_id,
    **kwargs,
):
    namespace, env_id = env_id.split("::", 1)

    if namespace not in register:
        raise ValueError(f"Unknown namespace {namespace}")

    env, env_params = register[namespace](env_id, **kwargs)

    return env, env_params
