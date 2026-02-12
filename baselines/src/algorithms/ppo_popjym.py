from functools import partial

import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate
from memorax.algorithms import PPO
from memorax.networks import (
    FFN, FeatureExtractor, GatedResidual, Network, PreNorm,
    Stack, heads,
)


def make(cfg, env, env_params):
    action_dim = env.action_space(env_params).n
    features = 256


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

    actor = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(action_dim=action_dim),
    )
    critic = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(),
    )
    lr_schedule = optax.linear_schedule(
        init_value=5e-5,
        end_value=0.0,
        transition_steps=cfg.total_timesteps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )
    return PPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor,
        critic_network=critic,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
