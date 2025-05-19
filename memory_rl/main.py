import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro
from flax import core, struct
from gymnax.wrappers.purerl import FlattenObservationWrapper

import wandb
from dqn import Args, make_dqn
from drqn_ import Args, make_drqn
from sac import Args, make_sac
from utils import LogWrapper


def main():
    args = tyro.cli(Args)

    # assert (
    #     args.train_frequency % args.num_envs == 0
    # ), f"train_frequency must be divisible by num_envs, but got {args.train_frequency} and {args.num_envs}"
    # assert (
    #     args.target_network_frequency % args.num_envs == 0
    # ), f"target_network_frequency must be divisible by num_envs, but got {args.target_network_frequency} and {args.num_envs}"
    # assert (
    #     args.target_network_frequency % args.train_frequency == 0
    # ), f"target_network_frequency must be divisible by train_frequency, but got {args.target_network_frequency} and {args.train_frequency}"
    #
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=vars(args),
        )

    key = jax.random.key(args.seed)

    import time

    # alg = make_drqn(args)
    alg = make_sac(args)
    key, state = alg.init(key)

    key, info = alg.evaluate(key, state, num_steps=args.num_evaluation_steps)
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
        start = time.time()
        (
            key,
            state,
            info,
        ) = alg.train(
            key,
            state,
            num_steps=args.num_epoch_steps,
        )
        jax.block_until_ready(info)
        print("SPS: ", args.num_epoch_steps / (time.time() - start))
        key, info = alg.evaluate(key, state, num_steps=args.num_evaluation_steps)
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
                },
                step=(epoch * args.num_epoch_steps),
            )


if __name__ == "__main__":
    main()
