from memorax.environments import (
    ale,
    brax,
    craftax,
    grimax,
    gxm,
    gymnax,
    gymnasium,
    jaxmarl,
    mujoco_playground,
    navix,
    pobax,
    popgym_arcade,
    popjym,
    pufferlib,
    xminigrid,
)

register = {
    "ale": ale.make,
    "brax": brax.make,
    "craftax": craftax.make,
    "grimax": grimax.make,
    "gymnax": gymnax.make,
    "gymnasium": gymnasium.make,
    "gxm": gxm.make,
    "jaxmarl": jaxmarl.make,
    "mujoco_playground": mujoco_playground.make,
    "navix": navix.make,
    "pobax": pobax.make,
    "popgym_arcade": popgym_arcade.make,
    "popjym": popjym.make,
    "pufferlib": pufferlib.make,
    "xminigrid": xminigrid.make,
}


def make(
    env_id,
    **kwargs,
) -> tuple:
    namespace, env_id = env_id.split("::", 1)

    if namespace not in register:
        raise ValueError(f"Unknown namespace {namespace}")

    env, env_params = register[namespace](env_id, **kwargs)

    return env, env_params
