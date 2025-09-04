import time
import hydra
from omegaconf import DictConfig, OmegaConf

import jax.numpy as jnp
from hydra.utils import instantiate

OmegaConf.register_new_resolver("eval", eval)


def log(logger, logger_state, state, info, *, prefix="evaluation", sps=0):
    import jax

    def masked_mean(data, mask):
        nominator = jnp.sum(data * mask, axis=tuple(range(1, data.ndim)))
        denominator = jnp.sum(mask, axis=tuple(range(1, data.ndim)))
        return nominator / denominator

    episodic_returns = masked_mean(info["returned_episode_returns"], info["returned_episode"])
    episodic_lengths = masked_mean(info["returned_episode_lengths"], info["returned_episode"])


    if "returned_episode_regret" in info:
        episodic_regret = masked_mean(info["returned_episode_regret"], info["returned_episode"])

    step = jax.device_get(state.step)
    if not isinstance(step, int):
        step = step[0].item()

    data = {
        f"{prefix}/episodic_returns": episodic_returns,
        f"{prefix}/episodic_lengths": episodic_lengths,
        **{
            f"{prefix}/{k}": v.mean(axis=tuple(range(1, v.ndim)))
            for k, v in info.items()
            if not (
                k.endswith("episodic_returns")
                or k.endswith("episodic_lengths")
                or k.endswith("returned_episode")
                or k.endswith("returned_episode_returns")
                or k.endswith("returned_episode_lengths")
                or k.endswith("returned_episode_regret")
                or k.endswith("timestep")
                or k.endswith("step_regret")
                or k.endswith("total_regret")
                or k.startswith("losses")
            )
        },
        **{k: v.mean(axis=tuple(range(1, v.ndim))) for k, v in info.items() if k.startswith("losses")},
    }

    if "returned_episode_regret" in info:
        data[f"{prefix}/episodic_regret"] = episodic_regret


    if sps:
        data["SPS"] = sps

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

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params, logger)

    # key, state = algorithm.init(key)
    keys, state = jax.vmap(algorithm.init)(keys)

    if cfg.environment.env_id == "Craftax-Symbolic-v1":
        max_steps_in_episode = 20_000
    else:
        max_steps_in_episode = env_params.max_steps_in_episode

    # key, transitions = algorithm.evaluate(key, state, max_steps_in_episode)
    keys, transitions = jax.vmap(algorithm.evaluate, in_axes=(0, 0, None))(
        keys, state, max_steps_in_episode
    )

    logger_state = log(logger, logger_state, state, transitions.info)
    logger.emit(logger_state)

    # key, state = algorithm.warmup(key, state, cfg.algorithm.learning_starts)
    keys, state = jax.vmap(algorithm.warmup, in_axes=(0, 0, None))(
        keys, state, cfg.algorithm.learning_starts
    )

    for i in range(0, cfg.total_timesteps, cfg.num_train_steps):
        start = time.perf_counter()
        # key, state, info = algorithm.train(key, state, cfg.num_train_steps)
        keys, state, info = jax.vmap(algorithm.train, in_axes=(0, 0, None))(
            keys, state, cfg.num_train_steps
        )
        end = time.perf_counter()

        sps = cfg.num_train_steps / (end - start) * cfg.num_seeds
        logger_state = log(
            logger, logger_state, state, info, prefix="training", sps=sps
        )
        # print(sps)

        if i % cfg.evaluate_every == 0:
            # key, transitions = algorithm.evaluate(key, state, max_steps_in_episode)
            keys, transitions = jax.vmap(algorithm.evaluate, in_axes=(0, 0, None))(
                keys, state, max_steps_in_episode
            )
            logger_state = log(logger, logger_state, state, transitions.info)
        logger_state = logger.emit(logger_state)

    logger.finish(logger_state)


if __name__ == "__main__":
    main()  # type: ignore
