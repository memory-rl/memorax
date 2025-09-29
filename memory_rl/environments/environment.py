from hydra.utils import instantiate


def make(cfg, key):
    env, env_params = instantiate(cfg)

    if env_params is not None and "env_params" in cfg:
        env_params = env_params.replace(**cfg["env_params"])

    for wrapper in cfg.get("wrappers", []):
        env = instantiate(wrapper, env=env)

    return env, env_params
