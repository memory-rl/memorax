import gymnax
import hydra
import jax
import wandb
from omegaconf import DictConfig, OmegaConf

from memory_rl import Algorithm, TMazeClassicActive, TMazeClassicPassive, make
from memory_rl.environments.environment import make as make_env
from memory_rl.utils import BraxGymnaxWrapper, NavixGymnaxWrapper
from popjaxrl.envs import make as make_popjax_env

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="memory_rl/conf", config_name="config")
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
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    key = jax.random.key(cfg.seed)

    env, env_params = make_env(cfg.environment)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params)

    key, state = algorithm.init(key)

    key, info = algorithm.evaluate(key, state, cfg.num_evaluation_steps)

    episodic_returns = info["returned_episode_returns"][info["returned_episode"]].mean()
    episodic_lengths = info["returned_episode_lengths"][info["returned_episode"]].mean()
    print(
        f"Timestep: {state.step} - Episodic Return: {episodic_returns} - Episodic Length: {episodic_lengths}"
    )

    if cfg.logger.track:
        info = {
            f"info/{key}": value.mean()
            for key, value in info.items()
            if key
            not in [
                "returned_episode",
                "returned_episode_returns",
                "returned_episode_lengths",
            ]
        }
        wandb.log(
            {
                "evaluation/episodic_return": episodic_returns,
                "evaluation/episodic_length": episodic_lengths,
                **info,
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
            info = {
                f"info/{key}": value.mean()
                for key, value in info.items()
                if key
                not in [
                    "returned_episode",
                    "returned_episode_returns",
                    "returned_episode_lengths",
                ]
            }
            wandb.log(
                {
                    "evaluation/episodic_return": episodic_returns,
                    "evaluation/episodic_length": episodic_lengths,
                    **info,
                },
                step=state.step,
            )


if __name__ == "__main__":
    main()
