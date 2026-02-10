import hydra
import jax
from hydra.utils import instantiate
from memorax.loggers import Logger
from omegaconf import OmegaConf

from src import algorithm, environment, profile


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    env, env_params = environment.make(**cfg.environment)
    agent = algorithm.make(cfg, env, env_params)

    logger = instantiate(cfg.logger)
    logger_state = logger.init(cfg=OmegaConf.to_container(cfg, resolve=True))

    if cfg.algorithm.name == "r2d2":
        key = jax.random.key(cfg.seed)
        key, state = agent.init(key)
        key, state = agent.warmup(key, state, agent.cfg.learning_starts)

        train = profile(agent.train, num_steps=cfg.num_train_steps)

        for _ in range(0, cfg.total_timesteps, cfg.num_train_steps):
            (key, state, transitions), SPS = train(key, state, cfg.num_train_steps)
            stats = Logger.get_episode_statistics(transitions, "training")
            losses = transitions.losses
            data = {**stats, **losses, "training/SPS": SPS}
            logger_state = logger.log(
                logger_state, data, step=state.step.item()
            )
            logger.emit(logger_state)

    else:
        key = jax.random.key(cfg.seed)
        keys = jax.random.split(key, cfg.num_seeds)

        init = jax.vmap(agent.init)
        train = profile(
            jax.vmap(agent.train, in_axes=(0, 0, None)),
            num_steps=cfg.num_train_steps,
        )
        get_episode_statistics = jax.vmap(
            Logger.get_episode_statistics, in_axes=(0, None)
        )

        def log(state, transitions, SPS, logger_state):
            stats = get_episode_statistics(transitions, "training")
            losses = jax.vmap(lambda transitions: transitions.losses)(transitions)
            data = {**stats, **losses, "training/SPS": SPS}
            logger_state = logger.log(
                logger_state, data, step=state.step[0].item()
            )
            logger.emit(logger_state)
            return logger_state

        keys, state = init(keys)
        for _ in range(0, cfg.total_timesteps, cfg.num_train_steps):
            (keys, state, transitions), SPS = train(
                keys, state, cfg.num_train_steps
            )
            logger_state = log(state, transitions, SPS, logger_state)

    logger.finish(logger_state)


if __name__ == "__main__":
    main()
