"""DD-PPO on CartPole with synchronized multi-GPU training.

DD-PPO (Decentralized Distributed PPO) trains a single model with gradients
averaged across all devices. Unlike the multi-GPU IPPO example which runs
independent agents, DD-PPO achieves true data parallelism with synchronized
parameters.

Key differences from multi-GPU IPPO:
- Single model trained across all GPUs (vs. independent models)
- Gradients synchronized via lax.pmean (vs. no synchronization)
- Linear scaling of effective batch size (num_envs * num_devices)
- Parameters stay identical across devices after each update
"""

import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from memorax.algorithms import DDPPO, DDPPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, SequenceModelWrapper, heads

total_timesteps = 500_000
num_train_steps = 10_000

seed = 0
num_devices = jax.local_device_count()
print(f"Found {num_devices} devices: {jax.devices()}")

num_envs = 64 // num_devices

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = DDPPOConfig(
    name="DDPPO",
    num_envs=num_envs,
    num_eval_envs=0,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

d_model = 128
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = SequenceModelWrapper(
    MLP(features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414))
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

agent = DDPPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
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
    [DashboardLogger(title=f"DDPPO CartPole ({num_devices} GPUs)", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg={**asdict(cfg), "num_devices": num_devices, "effective_batch_size": num_envs * num_devices})

# Verify device placement
print(f"\nDD-PPO: Training single model across {num_devices} GPUs")
print(f"  Envs per GPU: {num_envs}")
print(f"  Effective batch size: {num_envs * num_devices}\n")

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
