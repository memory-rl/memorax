from src.algorithms import ppo_brax, ppo_popjym, pqn_popjym, r2d2_popjym

register = {
    ("ppo", "brax"): ppo_brax.make,
    ("ppo", "popjym"): ppo_popjym.make,
    ("pqn", "popjym"): pqn_popjym.make,
    ("pqn", "popgym_arcade"): pqn_popjym.make,
    ("r2d2", "popjym"): r2d2_popjym.make,
}


def make(cfg, env, env_params):
    family = cfg.environment.get("family", cfg.environment.namespace)
    key = (cfg.algorithm.name, family)
    return register[key](cfg, env, env_params)
