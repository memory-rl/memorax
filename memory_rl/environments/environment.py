import gymnax
import hydra

from memory_rl.utils import (
    BraxGymnaxWrapper,
    LogWrapper,
    NavixGymnaxWrapper,
    PixelCraftaxEnvWrapper,
    PopGymWrapper,
)
from craftax import craftax_env
from popjaxrl.envs import make as make_popjaxrl_env

from memory_rl.environments.tmaze_env import make_tmaze_env


def make_brax(env_id):
    env = BraxGymnaxWrapper(env_id, backend="mjx")
    return env, None


def make_craftax(env_id):
    env = craftax_env.make_craftax_env_from_name(env_id, auto_reset=True)
    env_params = env.default_params

    # if "Pixels" in env_id:
    #     env = PixelCraftaxEnvWrapper(env)

    return env, env_params


def make_popjaxrl(env_id):
    env, env_params = make_popjaxrl_env(env_id)
    env = PopGymWrapper(env)
    env_params = env.default_params
    return env, env_params


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
    "tmaze": make_tmaze_env,
    "Craftax-Pixels-v1": make_craftax,
    "Craftax-Symbolic-v1": make_craftax,
    "Craftax-Classic-Symbolic-v1": make_craftax,
    "Craftax-Classic-Pixels-v1": make_craftax,
}


def make(cfg):
    env, env_params = register[cfg.env_id](cfg.env_id)

    if env_params is not None:
        env_params = env_params.replace(**cfg.get("parameters", {}))

    env = LogWrapper(env)
    for wrapper in cfg.get("wrappers", []):
        env = hydra.utils.instantiate(wrapper, env=env)

    return env, env_params
