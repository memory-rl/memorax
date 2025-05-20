import jax
import tyro

import wandb

# from dqn import Args, make_dqn

from drqn_cleanup import Args, make_drqn

# from sac import Args, make_sac
from algorithm import Algorithm


def main():
    args = tyro.cli(Args)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=vars(args),
        )

    key = jax.random.key(args.seed)

    # algorithm: Algorithm = make_dqn(args)
    algorithm: Algorithm = make_drqn(args)
    # alg: Algorithm = make_sac(args)
    key, state = algorithm.init(key)

    key, info = algorithm.evaluate(key, state, num_steps=args.num_evaluation_steps)
    episodic_return = info["returned_episode_returns"][info["returned_episode"]].mean()
    print(f"Timestep: {0}, Return: {episodic_return}")

    if args.track:
        wandb.log(
            {
                "episodic_return": episodic_return,
            },
            step=0,
        )

    for epoch in range(1, args.num_epochs):
        (
            key,
            state,
            info,
        ) = algorithm.train(
            key,
            state,
            num_steps=args.num_epoch_steps,
        )
        # actor_loss = info["actor_loss"].mean()
        # critic_loss = info["critic_loss"].mean()
        # entropy_loss = info["entropy_loss"].mean()

        key, info = algorithm.evaluate(key, state, num_steps=args.num_evaluation_steps)
        episodic_return = info["returned_episode_returns"][
            info["returned_episode"]
        ].mean()
        print(
            f"Timestep: {epoch*args.num_epoch_steps}, Return: {info['returned_episode_returns'][info['returned_episode']].mean()}"
        )

        if args.track:
            wandb.log(
                {
                    "episodic_return": episodic_return,
                    # "actor_loss": actor_loss,
                    # "critic_loss": critic_loss,
                    # "entropy_loss": entropy_loss,
                },
                step=(epoch * args.num_epoch_steps),
            )


if __name__ == "__main__":
    main()
