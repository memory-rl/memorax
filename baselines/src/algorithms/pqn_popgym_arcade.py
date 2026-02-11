from functools import partial

import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import (
    FFN, FeatureExtractor, GatedResidual, Network, PreNorm,
    Stack, heads,
)


def make(cfg, env, env_params):
    action_dim = env.action_space(env_params).n
    features = cfg.torso.features

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential([
            nn.Conv(64, kernel_size=(5, 5), strides=(2, 2)),
            nn.leaky_relu,
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(128, kernel_size=(3, 3), strides=(2, 2)),
            nn.leaky_relu,
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(256, kernel_size=(3, 3), strides=(2, 2)),
            nn.leaky_relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(1, 1)),
            nn.Conv(512, kernel_size=(1, 1), strides=(1, 1)),
            nn.leaky_relu,
            lambda x: x.reshape((x.shape[0], x.shape[1], -1)),
        ]),
        action_extractor=partial(jax.nn.one_hot, num_classes=action_dim),
        features=features,
    )

    blocks = [m for _ in range(2) for m in (
        GatedResidual(module=PreNorm(module=instantiate(cfg.torso))),
        GatedResidual(module=PreNorm(module=FFN(features=features, expansion_factor=4))),
    )]

    torso = Stack(blocks=tuple(blocks))

    network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.DiscreteQNetwork(action_dim=action_dim),
    )

    lr_schedule = optax.linear_schedule(
        init_value=cfg.optimizer.learning_rate,
        end_value=0.0,
        transition_steps=cfg.total_timesteps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )

    epsilon_schedule = optax.linear_schedule(
        init_value=1.0, end_value=0.05, transition_steps=0.25 * cfg.total_timesteps
    )

    return PQN(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=network,
        optimizer=optimizer,
        epsilon_schedule=epsilon_schedule,
    )
