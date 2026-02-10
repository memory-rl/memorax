from hydra.utils import instantiate
from memorax.environments import environment

from src.environments.poatar import POAsterix, POBreakout, POFreeway, POSpaceInvaders

_CUSTOM_ENVS = {
    "poatar": {
        "Breakout": POBreakout,
        "SpaceInvaders": POSpaceInvaders,
        "Freeway": POFreeway,
        "Asterix": POAsterix,
    },
}


def make(namespace, env_id, **kwargs):
    if namespace in _CUSTOM_ENVS:
        env = _CUSTOM_ENVS[namespace][env_id](**(kwargs.get("kwargs") or {}))
        env_params = env.default_params
    else:
        env_id = f"{namespace}::{env_id}"
        env, env_params = environment.make(env_id, **(kwargs.get("kwargs") or {}))

    if env_params is not None:
        env_params = env_params.replace(**kwargs.get("env_params", {}))

    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
