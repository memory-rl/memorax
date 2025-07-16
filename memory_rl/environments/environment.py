from dataclasses import field

import chex
import gymnax
import hydra
import jax
import jax.numpy as jnp
from gymnax.wrappers import FlattenObservationWrapper

from memory_rl.utils import BraxGymnaxWrapper, LogWrapper, NavixGymnaxWrapper
from popjaxrl.envs import make as make_popjaxrl


def make_brax(env_id):
    env = BraxGymnaxWrapper(env_id, backend="mjx")
    return env, None


def make_navix(env_id):
    env = NavixGymnaxWrapper(env_id)
    return env, None


register = {
    "CartPole-v1": gymnax.make,
    "Asterix-MinAtar": gymnax.make,
    "Breakout-MinAtar": gymnax.make,
    "SpaceInvaders-MinAtar": gymnax.make,
    "Pendulum-v1": gymnax.make,
    "MemoryChain-bsuite": gymnax.make,
    "UmbrellaChain-bsuite": gymnax.make,
    "ant": make_brax,
    "hopper": make_brax,
    "walker2d": make_brax,
    "AutoencodeEasy": make_popjaxrl,
    "AutoencodeHard": make_popjaxrl,
    "BattleshipEasy": make_popjaxrl,
    "BattleshipHard": make_popjaxrl,
    "ConcentrationEasy": make_popjaxrl,
    "ConcentrationHard": make_popjaxrl,
    "CountRecallEasy": make_popjaxrl,
    "CountRecallHard": make_popjaxrl,
    "HigherLowerEasy": make_popjaxrl,
    "HigherLowerHard": make_popjaxrl,
    "RepeatFirstEasy": make_popjaxrl,
    "RepeatFirstHard": make_popjaxrl,
    "StatelessCartpoleEasy": make_popjaxrl,
    "StatelessCartpoleHard": make_popjaxrl,
    "Navix-Crossings-S9N1-v0": make_navix,
    "Navix-Dist-Shift-1-v0": make_navix,
    "Navix-Doorkey-5x5-v0": make_navix,
    "Navix-Empty-5x5-v0": make_navix,
    "Navix-Four-Rooms-v0": make_navix,
    "Navix-Go-To-Door-5x5-v0": make_navix,
    "Navix-Key-Corridor-S3R1-v0": make_navix,
    "Navix-Lava-Gap-S5-v0": make_navix,
}


def make(cfg):
    env, env_params = register[cfg.env_id](cfg.env_id)

    if env_params is not None:
        env_params = env_params.replace(**cfg.get("parameters", {}))

    env = LogWrapper(env)
    for wrapper in cfg.get("wrappers", []):
        env = hydra.utils.instantiate(wrapper, env=env)
    return env, env_params
