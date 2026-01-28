from memorax.environments import (brax, craftax, gxm, gymnax, jaxmarl,
                                  mujoco, navix, popgym_arcade,
                                  popjym, xminigrid)

register = {
    "brax": brax.make,
    "craftax": craftax.make,
    "gymnax": gymnax.make,
    "gxm": gxm.make,
    "jaxmarl": jaxmarl.make,
    "mujoco": mujoco.make,
    "navix": navix.make,
    "popgym_arcade": popgym_arcade.make,
    "popjym": popjym.make,
    "xminigrid": xminigrid.make,
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
