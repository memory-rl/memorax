import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.algorithms.ppo import PPO
from memorax.networks import MLP, FeatureExtractor, Network, heads


def make(cfg, env, env_params):

    torso = instantiate(cfg.torso)

    feature_extractor = FeatureExtractor(
        observation_extractor=MLP(
            features=[256, 256],
            activation=nn.tanh,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        ),
        action_extractor=MLP(
            features=[64],
            activation=nn.tanh,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        ),
    )

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Gaussian(
            action_dim=env.action_space(env_params).shape[0],
            kernel_init=nn.initializers.orthogonal(0.01),
        ),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4, eps=1e-5),
    )

    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(
            kernel_init=nn.initializers.orthogonal(1.0),
        ),
    )

    agent = PPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
    return agent
