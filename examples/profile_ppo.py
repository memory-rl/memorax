import flax.linen as nn
import jax
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.networks import MLP, FeatureExtractor, Network, heads

TRACE_DIR = "/tmp/jax-trace-ppo"

NUM_TRAIN_STEPS = 1_000
NUM_WARMUP_STEPS = 500

seed = 0

print("Setting up environment and agent...", flush=True)

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = PPOConfig(
    name="PPO",
    num_envs=8,
    num_eval_envs=16,
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

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = MLP(features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414))
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
    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

key = jax.random.key(seed)

agent = PPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

print("Initializing agent...", flush=True)
key, state = agent.init(key)
print("Agent initialized.", flush=True)

print(f"Warming up (JIT compilation) with {NUM_WARMUP_STEPS} steps...", flush=True)
key, state, _ = agent.train(key, state, NUM_WARMUP_STEPS)
jax.block_until_ready(state)
print("Warmup complete.", flush=True)

print(f"Starting profiled training with {NUM_TRAIN_STEPS} steps...", flush=True)
print(f"Trace will be saved to: {TRACE_DIR}", flush=True)

with jax.profiler.trace(TRACE_DIR):
    print("Trace started, running training...", flush=True)
    key, state, transitions = agent.train(key, state, NUM_TRAIN_STEPS)
    jax.block_until_ready(state)
    print("Training complete, closing trace...", flush=True)

print("\nProfiling complete!")
print(f"\nTrace saved to: {TRACE_DIR}")
print("\nTo view the trace:")
print("  Option 1 - TensorBoard:")
print(f"    tensorboard --logdir={TRACE_DIR}")
print("    Then open http://localhost:6006 and go to the 'Profile' tab")
print("\n  Option 2 - Perfetto (recommended):")
print("    1. Open https://ui.perfetto.dev/")
print("    2. Click 'Open trace file'")
print(f"    3. Select the .json.gz file from {TRACE_DIR}/plugins/profile/")
