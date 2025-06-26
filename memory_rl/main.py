import gymnax
import hydra
import jax
from omegaconf import DictConfig, OmegaConf

import wandb
from memory_rl import Algorithm, TMazeClassicActive, TMazeClassicPassive, make
from memory_rl.utils import BraxGymnaxWrapper, NavixGymnaxWrapper
from popjaxrl.envs import make as make_popjax_env


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
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    key = jax.random.key(cfg.seed)

    if cfg.environment.env_id == "TMazeClassicActive":
        env = TMazeClassicActive()
        env_params = env.default_params
    elif cfg.environment.env_id == "TMazeClassicPassive":
        env = TMazeClassicPassive()
        env_params = env.default_params
    else:
        try:
            env, env_params = gymnax.make(cfg.environment.env_id)
        except ValueError:
            try:
                env = NavixGymnaxWrapper(cfg.environment.env_id)
                env_params = None
            except NotImplementedError:
                try:
                    env = BraxGymnaxWrapper(cfg.environment.env_id)
                    env_params = None
                except KeyError:
                    env, env_params = make_popjax_env(cfg.environment.env_id)

    algorithm: Algorithm = make(cfg.algorithm.name, cfg, env, env_params)

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
