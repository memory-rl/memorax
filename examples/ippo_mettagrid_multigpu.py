"""IPPO on MettaGrid arena environment with Multi-GPU support.

Uses jax.pmap to run independent training instances in parallel across GPUs.
Each GPU maintains its own environment and training state.

Requirements:
    pip install mettagrid
"""

import time
from dataclasses import asdict
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from memorax.algorithms import IPPO, IPPOConfig
from memorax.environments.mettagrid import (
    FlattenObservationWrapper,
    MettagridEnvironment,
)
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, SequenceModelWrapper, heads

total_timesteps = 100_000_000
num_train_steps = 25_000_000

seed = 0
num_devices = jax.local_device_count()
print(f"Found {num_devices} devices: {jax.devices()}")

from cogames.cogs_vs_clips.missions import make_cogsguard_mission

num_agents = 10
num_workers = 64 // num_devices
num_envs = 1024 // num_devices


def make_env():
    cfg = make_cogsguard_mission(num_agents=num_agents).make_env()
    return FlattenObservationWrapper(MettagridEnvironment(cfg, num_workers=num_workers))


envs = [make_env() for _ in range(num_devices)]

cfg = IPPOConfig(
    name="IPPO-MultiGPU",
    num_envs=num_envs,
    num_eval_envs=0,
    num_steps=256,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=1,
    normalize_advantage=True,
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

action_space = envs[0].action_spaces[envs[0].agents[0]]

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

# One agent per device
agents = [
    IPPO(
        cfg=cfg,
        env=envs[i],
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
    for i in range(num_devices)
]

# Initialize on each device
key = jax.random.key(seed)
device_keys = jax.random.split(key, num_devices)

keys, states = [], []
for i, device in enumerate(jax.devices()):
    with jax.default_device(device):
        k = jax.device_put(device_keys[i], device)
        k, s = agents[i].init(k)
        # Keep state on this device
        s = jax.device_put(s, device)
        keys.append(k)
        states.append(s)

# Logger
logger = Logger(
    [DashboardLogger(title=f"IPPO MettaGrid ({num_devices} GPUs)", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg={**asdict(cfg), "num_devices": num_devices})

# Verify device placement
for i, s in enumerate(states):
    device = s.step.devices()
    print(f"  Agent {i} state on: {device}")

print(f"\nTraining: {num_devices} GPUs, {num_envs} envs/GPU, {num_workers} workers/GPU\n")

for epoch in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()

    # Dispatch all devices (async, parallel)
    results = []
    for i, device in enumerate(jax.devices()):
        with jax.default_device(device):
            k, s, t = agents[i].train(keys[i], states[i], num_train_steps)
            results.append((k, jax.device_put(s, device), t))
    keys, states, transitions = zip(*results)
    keys, states, transitions = list(keys), list(states), list(transitions)

    jax.block_until_ready(states)
    end = time.perf_counter()

    SPS = (num_train_steps * num_devices) / (end - start)

    # Aggregate metrics across devices
    all_stats = [Logger.get_episode_statistics(t, "training") for t in transitions]
    stats = jax.tree.map(lambda *xs: sum(xs) / len(xs), *all_stats)

    all_losses = [jax.tree.map(lambda x: x.mean(), t.losses) for t in transitions]
    losses = jax.tree.map(lambda *xs: sum(xs) / len(xs), *all_losses)

    logger_state = logger.log(logger_state, {"training/SPS": SPS, **stats, **losses}, step=states[0].step.item())
    logger.emit(logger_state)

for e in envs:
    e.close()
