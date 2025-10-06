from gymnax.wrappers import FlattenObservationWrapper

from memory_rl.environments import (
    brax,
    craftax,
    gxm,
    gymnax,
    mujoco,
    navix,
    popgym_arcade,
    popjaxrl,
)

register = {
    "brax": brax.make,
    "craftax": craftax.make,
    "gymnax": gymnax.make,
    "gxm": gxm.make,
    "mujoco": mujoco.make,
    "navix": navix.make,
    "popgym_arcade": popgym_arcade.make,
    "popjaxrl": popjaxrl.make,
}


def make(cfg):

    if cfg.namespace not in register:
        raise ValueError(f"Unknown namespace {cfg.namespace}")

    env, env_params = register[cfg.namespace](cfg=cfg)

    if cfg.get("flatten_obs", False):
        env = FlattenObservationWrapper(env)

    if env_params is not None and "env_params" in cfg:
        env_params = env_params.replace(**cfg["env_params"])

    return env, env_params
