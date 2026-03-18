import jax.random

max_steps_in_episode = {
    "tmaze_5": 7,
    "tmaze_10": 12,
    "tmaze_20": 22,
    "tmaze_50": 52,
    "simple_chain": 10,
    "battleship": 100,
    "battleship_5": 25,
    "battleship_3": 9,
    "pocman": 1000,
    "ReacherPOMDP": 100,
    "rocksample_5_5": 25,
    "rocksample_7_8": 49,
    "rocksample_11_11": 121,
    "rocksample_15_15": 225,
    "fishing_0": 1000,
    "fishing_1": 1000,
    "fishing_2": 1000,
    "fishing_3": 1000,
    "fishing_4": 1000,
    "fishing_5": 1000,
    "fishing_6": 1000,
    "fishing_7": 1000,
    "fishing_8": 1000,
    "ant": 1000,
    "halfcheetah": 1000,
    "hopper": 1000,
    "walker2d": 1000,
    "craftax": 10000,
    "craftax_pixels": 10000,
    "craftax_classic": 10000,
    "craftax_classic_pixels": 10000,
}


def make(env_id: str, **kwargs) -> tuple:
    from pobax.envs import get_env
    from pobax.envs.wrappers.gymnax import (
        NormalizeVecObservation,
        NormalizeVecReward,
        VecEnv,
    )
    from pobax.envs.wrappers.observation import NamedObservationWrapper

    env, env_params = get_env(env_id, rand_key=jax.random.key(0), **kwargs)

    while isinstance(
        env,
        (VecEnv, NamedObservationWrapper, NormalizeVecObservation, NormalizeVecReward),
    ):
        env = env._env

    if env_id in max_steps_in_episode:
        env_params = env_params.replace(
            max_steps_in_episode=max_steps_in_episode[env_id]
        )

    return env, env_params
