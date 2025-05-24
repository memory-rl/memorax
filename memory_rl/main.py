import hydra
import jax
from omegaconf import DictConfig, OmegaConf

import wandb
from memory_rl import Algorithm, make


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.logger.track:
        name = (
            f"{cfg.algorithm.name}_{cfg.environment.env_id}_{cfg.seed}_{wandb.util.generate_id()}"
        ).lower()
        wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            name=name,
            group=f"{cfg.algorithm.name}_{cfg.environment.env_id}",
        )

    key = jax.random.key(cfg.seed)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg)

    key, state = algorithm.init(key)

    key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

    episodic_returns = info["returned_episode_returns"][info["returned_episode"]].mean()
    episodic_lengths = info["returned_episode_lengths"][info["returned_episode"]].mean()
    print(
        f"Timestep: {state.step} - Episodic Return: {episodic_returns} - Episodic Length: {episodic_lengths}"
    )

    if cfg.logger.track:
        wandb.log(
            {
                "evaluation/episodic_return": episodic_returns,
                "evaluation/episodic_length": episodic_lengths,
            },
            step=state.step,
        )

    key, state = algorithm.warmup(key, state, cfg.algorithm.learning_starts)
    for epoch in range(1, (cfg.total_timesteps // cfg.num_train_steps) + 1):
        key, state, info = algorithm.train(
            key,
            state,
            cfg.num_train_steps,
        )

        key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)
        episodic_returns = info["returned_episode_returns"][
            info["returned_episode"]
        ].mean()
        episodic_lengths = info["returned_episode_lengths"][
            info["returned_episode"]
        ].mean()
        print(
            f"Timestep: {state.step} - Episodic Return: {episodic_returns} - Episodic Length: {episodic_lengths}"
        )

        if cfg.logger.track:
            wandb.log(
                {
                    "evaluation/episodic_return": episodic_returns,
                    "evaluation/episodic_length": episodic_lengths,
                    # **{f"losses/{key}": value for key, value in info["losses"].items()},
                },
                step=state.step,
            )


if __name__ == "__main__":
    main()
