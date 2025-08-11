import jax
import gymnax
import hydra

from memory_rl.utils import BraxGymnaxWrapper, LogWrapper, NavixGymnaxWrapper, VecEnv
from popjaxrl.envs import make as make_popjaxrl
from pobax.envs import get_env
from memory_rl.environments.tmaze_env import make_tmaze_env



def make_brax(env_id):
    env = BraxGymnaxWrapper(env_id, backend="mjx")
    return env, None


def make_navix(env_id):
    env = NavixGymnaxWrapper(env_id)
    return env, None

def make_pobax(env_id, seed=0):
    key = jax.random.key(seed)
    env, env_params = get_env(env_id, key, apply_wrappers=False)
    jax.debug.breakpoint()
    return env, env_params


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
    "tmaze_10": make_pobax,
    "battleship_10": make_pobax,
}


def make(cfg):
    env, env_params = register[cfg.env_id](cfg.env_id)

    if env_params is not None:
        env_params = env_params.replace(**cfg.get("parameters", {}))

    env = LogWrapper(env)
    for wrapper in cfg.get("wrappers", []):
        env = hydra.utils.instantiate(wrapper, env=env)

    return env, env_params
