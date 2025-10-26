import jax
import flax.linen as nn
import optax
from memory_rl.algorithms.rppo import RPPO, RPPOConfig
from memory_rl.environments import environment
from memory_rl.networks import (
    MLP,
    SequenceNetwork,
    heads,
    FeatureExtractor,
    GTrXL,
)
from memory_rl.networks.recurrent.rnn import RNN

total_timesteps = 1_000_000
num_train_steps = 50_000
num_eval_steps = 5_000

# env, env_params = environment.make("gymnax::CartPole-v1")
env, env_params = environment.make("gymnax::MemoryChain-bsuite")

memory_length = 3
env_params = env_params.replace(
    memory_length=memory_length, max_steps_in_episode=memory_length + 1
)


cfg = RPPOConfig(
    name="rppo",
    learning_rate=3e-4,
    num_envs=1,
    num_eval_envs=1,
    num_steps=8,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=1,
    update_epochs=1,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_starts=0,
)

actor_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(128,))),
    # torso=GPT2(features=128, num_layers=1, num_heads=1, context_length=6),
    torso=GTrXL(
        features=128, num_layers=1, num_heads=1, context_length=2, memory_length=2
    ),
    # torso=S5(features=128, state_size=32, num_layers=4),
    # torso=FFM(features=128, memory_size=32, context_size=16),
    # torso=RNN(cell=nn.GRUCell(features=128)),
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
    ),
)
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

critic_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(128,))),
    # torso=GPT2(features=128, num_layers=1, num_heads=1, context_length=6),
    # torso=GTrXL(
    #     features=128, num_layers=4, num_heads=4, context_length=64, memory_length=64
    # ),
    # torso=FFM(features=128, memory_size=32, context_size=16),
    torso=RNN(cell=nn.GRUCell(features=128)),
    head=heads.VNetwork(),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

key = jax.random.key(1)

agent = RPPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor=actor_network,
    critic=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
)

key, state = agent.init(key)

print("================ FINISHED INITIALIZATION ================")

key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)

for i in range(0, total_timesteps, num_train_steps):

    key, state, transitions = agent.train(key, state, num_steps=num_train_steps)
    # key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
