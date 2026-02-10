from functools import partial

import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate
from memorax.algorithms import R2D2
from memorax.buffers import make_prioritised_episode_buffer
from memorax.networks import (
    FFN, FeatureExtractor, GatedResidual, Network, PreNorm,
    Stack, heads,
)


def make(cfg, env, env_params):
    action_dim = env.action_space(env_params).n
    features = cfg.torso.features


    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential([nn.Dense(features), nn.LayerNorm(), nn.relu]),
        action_extractor=partial(jax.nn.one_hot, num_classes=action_dim),
        features=features,
    )

    blocks = [m for _ in range(2) for m in (
        GatedResidual(module=PreNorm(module=instantiate(cfg.torso))),
        GatedResidual(module=PreNorm(module=FFN(features=features, expansion_factor=4))),
    )]

    torso = Stack(blocks=tuple(blocks))

    q_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.DiscreteQNetwork(action_dim=action_dim),
    )

    algo_cfg = instantiate(cfg.algorithm)

    lr_schedule = optax.linear_schedule(
        init_value=cfg.optimizer.learning_rate,
        end_value=0.0,
        transition_steps=cfg.total_timesteps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule),
    )

    buffer = make_prioritised_episode_buffer(
        max_length=algo_cfg.buffer_size,
        min_length=algo_cfg.learning_starts,
        sample_batch_size=algo_cfg.batch_size,
        sample_sequence_length=algo_cfg.burn_in_length + algo_cfg.sequence_length,
        add_batch_size=algo_cfg.num_envs,
        priority_exponent=algo_cfg.priority_exponent,
    )

    epsilon_schedule = optax.linear_schedule(
        init_value=algo_cfg.start_e,
        end_value=algo_cfg.end_e,
        transition_steps=int(algo_cfg.exploration_fraction * cfg.total_timesteps),
    )
    beta_schedule = optax.linear_schedule(
        init_value=algo_cfg.importance_sampling_exponent,
        end_value=1.0,
        transition_steps=cfg.total_timesteps,
    )

    return R2D2(
        cfg=algo_cfg,
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
        beta_schedule=beta_schedule,
    )
