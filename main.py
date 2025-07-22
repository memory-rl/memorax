import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from memory_rl import Algorithm, make
from memory_rl.environments.environment import make as make_env

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="memory_rl/conf", config_name="config")
def main(cfg: DictConfig):

    logger = instantiate(cfg.logger)
    logger.init(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.key(cfg.seed)

    env, env_params = make_env(cfg.environment)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params)

    key, state = algorithm.init(key)

    key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

    episodic_returns = info["returned_episode_returns"][info["returned_episode"]].mean()
    episodic_lengths = info["returned_episode_lengths"][info["returned_episode"]].mean()
    logger.log(
        {"episodic_returns": episodic_returns, "episodic_lengths": episodic_lengths},
        step=state.step,
    )

    key, state = algorithm.warmup(key, state, cfg.algorithm.learning_starts)

    def cond_fn(carry):
        _, state = carry
        return state.step < cfg.total_timesteps

    def body_fn(carry):
        key, state = carry
        key, state, info = algorithm.train(key, state, cfg.num_train_steps)
        key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

        def callback(info, step):
            episodic_returns = info["returned_episode_returns"][
                info["returned_episode"]
            ].mean()
            episodic_lengths = info["returned_episode_lengths"][
                info["returned_episode"]
            ].mean()
            data = {
                "episodic_returns": episodic_returns,
                "episodic_lengths": episodic_lengths,
            }
            logger.log(data, step=step)

        jax.debug.callback(callback, info, state.step)

        return (key, state)

    jax.lax.while_loop(cond_fn, body_fn, (key, state))


if __name__ == "__main__":
    main()  # type: ignore
