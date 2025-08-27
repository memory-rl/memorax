import hydra
import jax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from memory_rl import Algorithm, make
from memory_rl.environments.environment import make as make_env

OmegaConf.register_new_resolver("eval", eval)


def log(logger, logger_state, state, info, *, prefix="evaluation"):
    info = jax.device_get(info)
    step = jax.device_get(state.step).item()
    episodic_returns = info["returned_episode_returns"][info["returned_episode"]].mean()
    episodic_lengths = info["returned_episode_lengths"][info["returned_episode"]].mean()
    data = {
        f"{prefix}/episodic_returns": episodic_returns,
        f"{prefix}/episodic_lengths": episodic_lengths,
        **{
            f"{prefix}/{k}": v.mean()
            for k, v in info.items()
            if not (
                k.endswith("episodic_returns")
                or k.endswith("episodic_lengths")
                or k.endswith("returned_episode")
                or k.endswith("returned_episode_returns")
                or k.endswith("returned_episode_lengths")
                or k.startswith("losses")
            )
        },
        **{k: v.mean() for k, v in info.items() if k.startswith("losses")},
    }
    logger_state = logger.log(logger_state, data, step=step)

    return logger_state


@hydra.main(version_base=None, config_path="memory_rl/conf", config_name="config")
def main(cfg: DictConfig):

    logger = instantiate(cfg.logger)
    logger_state = logger.init(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.key(cfg.seed)

    env, env_params = make_env(cfg.environment)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params, logger)

    key, state = algorithm.init(key)

    key, transitions = algorithm.evaluate(key, state, env_params.max_steps_in_episode)

    logger_state = log(logger, logger_state, state, transitions.info)
    logger.emit(logger_state)

    key, state = algorithm.warmup(key, state, cfg.algorithm.learning_starts)

    while state.step < cfg.total_timesteps:
        key, state, info = algorithm.train(key, state, cfg.num_train_steps)
        logger_state = log(logger, logger_state, state, info, prefix="training")
        key, transitions = algorithm.evaluate(
            key, state, env_params.max_steps_in_episode
        )
        logger_state = log(logger, logger_state, state, transitions.info)
        logger_state = logger.emit(logger_state)

    logger.finish(logger_state)


if __name__ == "__main__":
    main()  # type: ignore
