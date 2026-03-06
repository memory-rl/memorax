import jax.random


def make(env_id, **kwargs):
    from pobax.envs import get_env
    from pobax.envs.wrappers.gymnax import (
        NormalizeVecObservation,
        NormalizeVecReward,
        VecEnv,
    )
    from pobax.envs.wrappers.observation import NamedObservationWrapper

    env, env_params = get_env(env_id, rand_key=jax.random.PRNGKey(0), **kwargs)

    while isinstance(
        env,
        (VecEnv, NamedObservationWrapper, NormalizeVecObservation, NormalizeVecReward),
    ):
        env = env._env

    return env, env_params
