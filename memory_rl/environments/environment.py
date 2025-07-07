from dataclasses import field

import chex
import gymnax
import hydra
import jax
import jax.numpy as jnp
from gymnax.wrappers import FlattenObservationWrapper

from memory_rl.utils import BraxGymnaxWrapper, LogWrapper, NavixGymnaxWrapper
from popjaxrl.envs import make as make_popjaxrl

register = {
    "CartPole-v1": gymnax.make,
    "Asterix-MinAtar": gymnax.make,
    "Breakout-MinAtar": gymnax.make,
    "SpaceInvaders-MinAtar": gymnax.make,
    "Pendulum-v1": gymnax.make,
    "MemoryChain-bsuite": gymnax.make,
    "UmbrellaChain-bsuite": gymnax.make,
    "ant": BraxGymnaxWrapper,
    "hopper": BraxGymnaxWrapper,
    "walker2d": BraxGymnaxWrapper,
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
    "Navix-Crossings-S9N1-v0": NavixGymnaxWrapper,
    "Navix-Dist-Shift-1-v0": NavixGymnaxWrapper,
    "Navix-Doorkey-5x5-v0": NavixGymnaxWrapper,
    "Navix-Empty-5x5-v0": NavixGymnaxWrapper,
    "Navix-Four-Rooms-v0": NavixGymnaxWrapper,
    "Navix-Go-To-Door-5x5-v0": NavixGymnaxWrapper,
    "Navix-Key-Corridor-S3R1-v0": NavixGymnaxWrapper,
    "Navix-Lava-Gap-S5-v0": NavixGymnaxWrapper,
}


def make(cfg):
    env, env_params = register[cfg.env_id](cfg.env_id)
    env_params = env_params.replace(**cfg.get("parameters", {}))

    env = LogWrapper(env)
    for wrapper in cfg.get("wrappers", []):
        env = hydra.utils.instantiate(wrapper, env=env)
    return env, env_params
