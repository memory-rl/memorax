import hydra
import jax
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from memory_rl import Algorithm, make
from memory_rl.environments.environment import make as make_env
from memory_rl.utils import callback

OmegaConf.register_new_resolver("eval", eval)

@callback
def log(logger, info, step):
    episodic_returns = info["returned_episode_returns"][
        info["returned_episode"]
    ].mean()
    episodic_lengths = info["returned_episode_lengths"][
        info["returned_episode"]
    ].mean()
    data = {
        "evaluation/episodic_returns": episodic_returns,
        "evaluation/episodic_lengths": episodic_lengths,
    }
    logger.log(data, step=step)


@hydra.main(version_base=None, config_path="memory_rl/conf", config_name="config")
def main(cfg: DictConfig):
    
    logger = instantiate(cfg.logger)
    logger.init(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.key(cfg.seed)

    env, env_params = make_env(cfg.environment)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params, logger)

    key, state = algorithm.init(key)

    key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

    log(logger, info, state.step)

    key, state = algorithm.warmup(key, state, cfg.algorithm.learning_starts)

    def cond_fn(carry):
        _, state = carry
        return state.step < cfg.total_timesteps

    def body_fn(carry):
        key, state = carry
        key, state, info = algorithm.train(key, state, cfg.num_train_steps)
        key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

        log(logger, info, state.step)

        return (key, state)

    jax.lax.while_loop(cond_fn, body_fn, (key, state))
    logger.finish()



if __name__ == "__main__":
    main()  # type: ignore
