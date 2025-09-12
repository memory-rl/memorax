import gymnax
import hydra

from memory_rl.utils import (
    BraxGymnaxWrapper,
    LogWrapper,
    NavixGymnaxWrapper,
    PixelCraftaxEnvWrapper,
    PopGymWrapper,
    GxmGymnaxWrapper,
)
from craftax import craftax_env
from popjaxrl.envs import make as make_popjaxrl_env
import gxm

from memory_rl.environments.tmaze_env import make_tmaze_env

def make_gymnax(env_id, key):
    env, env_params = gymnax.make(env_id)
    return env, env_params

def make_gxm(env_id, key):
    env = gxm.make(env_id)
    env = GxmGymnaxWrapper(env, key)
    env_params = env.default_params
    return env, env_params


def make_brax(env_id, key):
    env = BraxGymnaxWrapper(env_id, backend="mjx")
    return env, None


def make_craftax(env_id, key):
    env = craftax_env.make_craftax_env_from_name(env_id, auto_reset=True)
    env_params = env.default_params

    # if "Pixels" in env_id:
    #     env = PixelCraftaxEnvWrapper(env)

    return env, env_params


def make_popjaxrl(env_id, key):
    env, env_params = make_popjaxrl_env(env_id)
    env = PopGymWrapper(env)
    env_params = env.default_params
    return env, env_params


def make_navix(env_id, key):
    env = NavixGymnaxWrapper(env_id)
    return env, None


register = {
    "CartPole-v1": make_gymnax,
    "Asterix-MinAtar": make_gymnax,
    "Breakout-MinAtar": make_gymnax,
    "SpaceInvaders-MinAtar": make_gymnax,
    "Pendulum-v1": make_gymnax,
    "MemoryChain-bsuite": make_gymnax,
    "UmbrellaChain-bsuite": make_gymnax,
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
    "Envpool/Breakout-v5": make_gxm,
    "Gymnax/CartPole-v1": make_gxm,
}


def make(cfg, key):
    env, env_params = register[cfg.env_id](cfg.env_id, key)

    if env_params is not None:
        env_params = env_params.replace(**cfg.get("parameters", {}))

    # env = LogWrapper(env)
    for wrapper in cfg.get("wrappers", []):
        env = hydra.utils.instantiate(wrapper, env=env)

    return env, env_params
