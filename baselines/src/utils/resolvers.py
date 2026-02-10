from pathlib import Path

import gymnax
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import environment


def get_action_dim(cfg):
    env, env_params = environment.make(**cfg)

    if isinstance(env.action_space(env_params), gymnax.environments.spaces.Discrete):
        action_dim = env.action_space(env_params).n
    else:

        action_dim, *_ = env.action_space(env_params).shape
    return action_dim


def cascading_fallback(algorithm: str, environment: str, torso=None) -> str:
    gh = GlobalHydra.instance()

    loader = gh.config_loader()

    group, env_id = environment.split("/", 1)

    search_group = f"hyperparameters/{algorithm}/{group}"

    if torso:

        available_options = loader.get_group_options(f"{search_group}/{env_id}")

        if torso in available_options:
            return f"{algorithm}/{group}/{env_id}/{torso}"

    search_group = f"hyperparameters/{algorithm}/{group}"
    available_options = loader.get_group_options(search_group)

    if env_id in available_options:
        return f"{algorithm}/{group}/{env_id}"

    return f"{algorithm}/{group}"


def get_group(_root_):
    group = f"{_root_.algorithm.name}_{_root_.environment.namespace}_{_root_.environment.env_id}_{'_'.join(f'{k}_{v}' for k, v in sorted(_root_.environment.get('kwargs', {}).items()))}"
    return group


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_action_dim", get_action_dim)
OmegaConf.register_new_resolver("cascading_fallback", cascading_fallback)
OmegaConf.register_new_resolver("get_group", get_group)
