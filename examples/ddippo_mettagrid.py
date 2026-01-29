"""DD-IPPO on MettaGrid arena environment with synchronized multi-GPU training.

DD-IPPO (Decentralized Distributed Independent PPO) trains a single model with
gradients averaged across all devices. Unlike the multi-GPU IPPO example which
runs independent agents, DD-IPPO achieves true data parallelism with synchronized
parameters.

Key differences from multi-GPU IPPO:
- Single model trained across all GPUs (vs. independent models)
- Gradients synchronized via lax.pmean (vs. no synchronization)
- Linear scaling of effective batch size (num_envs * num_devices)
- Parameters stay identical across devices after each update

Requirements:
    pip install mettagrid
"""

import os
import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from memorax.algorithms import DDIPPO, DDIPPOConfig
from memorax.environments.mettagrid import (
    FlattenObservationWrapper,
    MettagridEnvironment,
)
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, SequenceModelWrapper, heads


def main():
    total_timesteps = 100_000_000
    num_train_steps = 25_000_000

    seed = 0
    num_devices = jax.local_device_count()
    print(f"Found {num_devices} devices: {jax.devices()}")

    from cogames.cogs_vs_clips.missions import make_cogsguard_mission

    num_agents = 10
    # Split envs across devices for linear batch scaling
    num_envs = 1024 // num_devices
    # Respect hardware core limits and ensure num_envs is divisible by num_workers
    num_cores = os.cpu_count() or 1
    max_workers = min(64 // num_devices, num_cores)
    # Find largest divisor of num_envs that doesn't exceed max_workers
    num_workers = max(w for w in range(1, max_workers + 1) if num_envs % w == 0)

    env_cfg = make_cogsguard_mission(num_agents=num_agents).make_env()
    env = FlattenObservationWrapper(MettagridEnvironment(env_cfg, num_workers=num_workers))

    cfg = DDIPPOConfig(
        name="DDIPPO",
        num_envs=num_envs,
        num_eval_envs=0,
        num_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=8,
        update_epochs=1,
        normalize_advantage=True,  # Global normalization across all devices
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    d_model = 256
    feature_extractor = FeatureExtractor(
        observation_extractor=MLP(
            features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    )
    torso = SequenceModelWrapper(
        MLP(features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414))
    )

    action_space = env.action_spaces[env.agents[0]]

    VmappedNetwork = nn.vmap(
        Network,
        variable_axes={"params": None},
        split_rngs={"params": False, "memory": True, "dropout": True},
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )

    actor_network = VmappedNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(
            action_dim=action_space.n,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        ),
    )

    critic_network = VmappedNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.muon(learning_rate=0.005, beta=0.95, nesterov=True),
    )

    # Single agent for distributed training
    agent = DDIPPO(
        cfg=cfg,
        env=env,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )

    # Initialize on single device, then replicate
    key = jax.random.key(seed)
    key, state = agent.init(key)

    # Replicate state across all devices for pmap
    keys, states = agent.replicate(key, state)

    # Logger
    logger = Logger(
        [DashboardLogger(title=f"DD-IPPO MettaGrid ({num_devices} GPUs)", total_timesteps=total_timesteps)]
    )
    logger_state = logger.init(cfg={**asdict(cfg), "num_devices": num_devices, "effective_batch_size": num_envs * num_devices})

    # Verify device placement
    print(f"\nDD-IPPO: Training single model across {num_devices} GPUs")
    print(f"  Envs per GPU: {num_envs}")
    print(f"  Effective batch size: {num_envs * num_devices}")
    print(f"  Workers per GPU: {num_workers}\n")

    for epoch in range(0, total_timesteps, num_train_steps):
        start = time.perf_counter()

        # Distributed training - gradients synchronized across devices
        keys, states, transitions = agent.train(keys, states, num_train_steps)
        jax.block_until_ready(states)

        end = time.perf_counter()

        # SPS accounts for all devices
        SPS = (num_train_steps * num_devices) / (end - start)

        # Get metrics from first device (all devices have synchronized params)
        transitions_0 = jax.tree.map(lambda x: x[0], transitions)
        training_statistics = Logger.get_episode_statistics(transitions_0, "training")
        losses = jax.tree.map(lambda x: x.mean(), transitions_0.losses)

        # Log step from first device (all identical due to sync)
        step = states.step[0].item()
        logger_state = logger.log(logger_state, {"training/SPS": SPS, **training_statistics, **losses}, step=step)
        logger.emit(logger_state)

    env.close()


if __name__ == "__main__":
    main()
