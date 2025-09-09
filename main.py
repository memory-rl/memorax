import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from memory_rl.loggers.logger import Logger

OmegaConf.register_new_resolver("eval", eval)


def log(logger, logger_state, step, transitions, gamma, *, prefix="evaluation", sps=0):
    import jax

    losses = jax.vmap(Logger.get_losses)(transitions)

    episode_statistics = jax.vmap(
        Logger.get_episode_statistics, in_axes=(0, None, None)
    )(transitions, gamma, prefix)

    data = {**losses, **episode_statistics}

    if sps:
        data["SPS"] = sps

    data = jax.device_get(data)
    logger_state = logger.log(logger_state, data, step=step)

    return logger_state


@hydra.main(version_base=None, config_path="memory_rl/conf", config_name="config")
def main(cfg: DictConfig):

    import jax

    from memory_rl import Algorithm, make
    from memory_rl.environments.environment import make as make_env

    logger = instantiate(cfg.logger)
    logger_state = logger.init(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.key(cfg.seed)
    keys = jax.random.split(key, cfg.num_seeds)

    env, env_params = make_env(cfg.environment)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params)

    keys, state = jax.vmap(algorithm.init)(keys)

    if cfg.environment.env_id == "Craftax-Symbolic-v1":
        max_steps_in_episode = 20_000
    else:
        max_steps_in_episode = env_params.max_steps_in_episode

    keys, transitions = jax.vmap(algorithm.evaluate, in_axes=(0, 0, None))(
        keys, state, max_steps_in_episode
    )

    log(
        logger,
        logger_state,
        0,
        transitions,
        cfg.algorithm.gamma,
        prefix="evaluation",
    )
    logger.emit(logger_state)

    keys, state = jax.vmap(algorithm.warmup, in_axes=(0, 0, None))(
        keys, state, cfg.algorithm.learning_starts
    )

    for i in range(0, cfg.total_timesteps, cfg.num_train_steps):
        start = time.perf_counter()
        keys, state, transitions = jax.vmap(algorithm.train, in_axes=(0, 0, None))(
            keys, state, cfg.num_train_steps
        )
        end = time.perf_counter()

        sps = cfg.num_train_steps / (end - start) * cfg.num_seeds
        logger_state = log(
            logger,
            logger_state,
            i,
            transitions,
            cfg.algorithm.gamma,
            prefix="training",
            sps=sps,
        )

        if i % cfg.evaluate_every == 0:
            keys, transitions = jax.vmap(algorithm.evaluate, in_axes=(0, 0, None))(
                keys, state, max_steps_in_episode
            )
            logger_state = log(
                logger, logger_state, i, transitions, cfg.algorithm.gamma
            )
        logger_state = logger.emit(logger_state)

    logger.finish(logger_state)


if __name__ == "__main__":
    main()  # type: ignore
