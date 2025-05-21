import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import chex
import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from gymnax.wrappers import FlattenObservationWrapper

from utils import LogWrapper, RecordEpisodeStatistics, compute_gae

# from popjaxrl.envs import make
# from popjaxrl.envs.wrappers import AliasPrevActionV2


@dataclass(frozen=True)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_seeds: int = 1
    """the number of seeds to run the experiment with"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "memory_rl"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    debug: bool = False
    """if toggled, this experiment will run in debug mode"""

    # Algorithm specific arguments
    env_id: str = "SpaceInvaders-MinAtar"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    learning_starts: int = 0
    """the number of steps before learning starts"""
    num_train_steps: int = 100_000
    """the number of steps per epoch"""
    num_evaluation_steps: int = 50_000

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def num_epochs(self):
        return self.total_timesteps // self.num_train_steps + 1


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        probs = distrax.Categorical(logits=logits)

        return probs


class Critic(nn.Module):

    @nn.compact
    def __call__(self, x: chex.Array):
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)

        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return value.squeeze(-1)


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class PPOState:
    step: int
    obs: chex.Array
    env_state: chex.Array
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    network_updates: int


@chex.dataclass(frozen=True)
class PPO:
    args: Args
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    actor: Actor
    critic: Critic
    optimizer: optax.GradientTransformation

    def create(self, key):
        key, env_key, actor_key, critic_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )

        actor_params = self.actor.init(actor_key, obs)
        actor_optimizer_state = self.optimizer.init(actor_params)

        critic_params = self.critic.init(critic_key, obs)
        critic_optimizer_state = self.optimizer.init(critic_params)

        return (
            key,
            PPOState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                actor_params=actor_params,  # type: ignore
                actor_optimizer_state=actor_optimizer_state,  # type: ignore
                critic_params=critic_params,  # type: ignore
                critic_optimizer_state=critic_optimizer_state,  # type: ignore
                network_updates=0,  # type: ignore
            ),
        )

    def warmup(self, key, state, num_steps):
        """No warmup needed for PPO"""
        return key, state

    def train(self, key, state, num_steps):

        def update_step(carry: tuple, _):

            def step(carry: tuple, _):
                key, state = carry

                key, action_key, step_key = jax.random.split(key, 3)

                probs = self.actor.apply(state.actor_params, state.obs)
                action = probs.sample(seed=action_key)
                log_prob = probs.log_prob(action)

                value = self.critic.apply(state.critic_params, state.obs)

                step_key = jax.random.split(step_key, self.args.num_envs)
                next_obs, env_state, reward, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(step_key, state.env_state, action, self.env_params)

                transition = Transition(
                    observation=state.obs,  # type: ignore
                    action=action,  # type: ignore
                    reward=reward,  # type: ignore
                    done=done,  # type: ignore
                    info=info,  # type: ignore
                    log_prob=log_prob,  # type: ignore
                    value=value,  # type: ignore
                )

                state = state.replace(
                    step=state.step + self.args.num_envs,
                    obs=next_obs,
                    env_state=env_state,
                )
                carry = (
                    key,
                    state,
                )
                return carry, transition

            (key, state), transitions = jax.lax.scan(
                step,
                carry,
                length=self.args.num_steps,
            )
            final_value = self.critic.apply(state.critic_params, state.obs)

            advantages, returns = compute_gae(
                self.args.gamma, self.args.gae_lambda, final_value, transitions
            )

            if self.args.norm_adv:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            batch = (transitions, advantages, returns)

            def update_epoch(carry: tuple, _):
                key, state, batch = carry

                def update_minibatch(state, minibatch: tuple):
                    transitions, advantages, returns = minibatch

                    def actor_loss_fn(params, transitions, advantages):
                        probs = self.actor.apply(params, transitions.observation)
                        log_prob = probs.log_prob(transitions.action)
                        entropy = probs.entropy().mean()

                        ratio = jnp.exp(log_prob - transitions.log_prob)
                        actor_loss = -jnp.minimum(
                            ratio * advantages,
                            jnp.clip(
                                ratio,
                                1.0 - self.args.clip_coef,
                                1.0 + self.args.clip_coef,
                            )
                            * advantages,
                        ).mean()
                        return actor_loss - self.args.ent_coef * entropy

                    def critic_loss_fn(params, transitions, advantages, returns):
                        value = self.critic.apply(params, transitions.observation)

                        if self.args.clip_vloss:
                            critic_loss = jnp.square(value - returns)
                            clipped_value = transitions.value + jnp.clip(
                                (value - transitions.value),
                                -self.args.clip_coef,
                                self.args.clip_coef,
                            )
                            clipped_critic_loss = jnp.square(clipped_value - returns)
                            critic_loss = (
                                0.5
                                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
                            )
                        else:
                            critic_loss = 0.5 * jnp.square(value - returns).mean()

                        return self.args.vf_coef * critic_loss

                    actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
                        state.actor_params, transitions, advantages
                    )
                    actor_updates, actor_optimizer_state = self.optimizer.update(
                        actor_grads, state.actor_optimizer_state, state.actor_params
                    )
                    acotr_params = optax.apply_updates(
                        state.actor_params, actor_updates
                    )

                    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                        state.critic_params, transitions, advantages, returns
                    )
                    critic_updates, critic_optimizer_state = self.optimizer.update(
                        critic_grads, state.critic_optimizer_state, state.critic_params
                    )
                    critic_params = optax.apply_updates(
                        state.critic_params, critic_updates
                    )

                    state = state.replace(
                        actor_params=acotr_params,
                        actor_optimizer_state=actor_optimizer_state,
                        critic_params=critic_params,
                        critic_optimizer_state=critic_optimizer_state,
                        network_updates=state.network_updates + 1,
                    )
                    return state, actor_loss + critic_loss

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(
                    permutation_key, self.args.batch_size
                )
                flattened_batch = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), flattened_batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [self.args.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                state, loss = jax.lax.scan(
                    update_minibatch,
                    state,
                    minibatches,
                )
                return (
                    key,
                    state,
                    batch,
                ), loss

            (key, state, batch), loss = jax.lax.scan(
                update_epoch,
                (key, state, batch),
                length=self.args.update_epochs,
            )
            transitions, *_ = batch

            if self.args.debug:

                def callback(info, loss):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * self.args.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(
                    callback,
                    transitions.info,
                    loss,
                )

            return (key, state), transitions.info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=num_steps // (self.args.num_envs * self.args.num_steps),
        )

        return key, state, info

    def evaluate(self, key, state, num_steps):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        def step(carry, _):
            key, obs, env_state = carry

            probs = self.actor.apply(state.actor_params, obs)
            action = jnp.argmax(probs.logits, axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.args.num_envs)
            obs, env_state, _, _, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, env_state), info

        (key, obs, env_state), info = jax.lax.scan(
            step, (key, obs, env_state), length=num_steps
        )

        return key, info


def make_ppo(args: Args):

    env, env_params = gymnax.make(args.env_id)
    env = LogWrapper(env)
    env = FlattenObservationWrapper(env)

    actor = Actor(action_dim=env.action_space(env_params).n)
    critic = Critic()

    if args.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=0.0,
            transition_steps=(args.total_timesteps // (args.num_envs * args.num_steps))
            * args.update_epochs
            * args.num_minibatches,
        )
    else:
        learning_rate = args.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    return PPO(
        args=args,
        env=env,
        env_params=env_params,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )
