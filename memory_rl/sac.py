import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from gymnax.wrappers.purerl import FlattenObservationWrapper

import wandb
from networks import MaskedLSTMCell, MaskedOptimizedLSTMCell, MaskedRNN
from utils import BraxGymnaxWrapper, LogWrapper, make_trajectory_buffer


@dataclass(frozen=True)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_seeds: int = 1
    """the number of increasing seeds"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "memory_rl"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    debug: bool = True
    """whether to print debug information"""
    plot: bool = False
    """whether to plot the returns"""

    # Algorithm specific arguments
    env_id: str = "ant"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    sample_sequence_length: int = 4
    """the length of the sequence to sample from the replay memory"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the target network update rate"""
    policy_frequency: int = 2
    """the timesteps it takes to update the policy"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    entropy_coefficient: float = 0.2
    """the entropy coefficient"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    num_epoch_steps: int = 50_000
    """the number of steps for each epoch"""
    num_evaluation_steps: int = 5_000
    """the number of steps for each evaluation"""

    @property
    def num_updates(self):
        return self.total_timesteps // self.num_envs

    @property
    def num_epochs(self):
        return self.total_timesteps // self.num_epoch_steps


class SoftQNetwork(nn.Module):

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        action: jnp.ndarray,
    ):
        x = jnp.concatenate([x, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        q_values = nn.Dense(1)(x)

        return q_values


class Actor(nn.Module):
    action_dim: int
    action_scale: float
    action_bias: float
    LOG_STD_MIN: float = -20
    LOG_STD_MAX: float = 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )
        std = jnp.exp(log_std)

        probs = distrax.Normal(mean, std)

        return probs

    def get_action(self, key, probs):
        raw_action = probs.sample(seed=key)
        squashed_action = jnp.tanh(raw_action)
        action = self.action_scale * squashed_action + self.action_bias
        log_prob = probs.log_prob(raw_action)
        log_prob -= jnp.log(self.action_scale * (1 - squashed_action**2) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)
        return action, log_prob


@chex.dataclass(frozen=True)
class SAC:
    init: Callable
    train: Callable
    evaluate: Callable


@chex.dataclass(frozen=True)
class SACState:
    step: int
    obs: chex.Array
    env_state: Any
    q1_params: core.FrozenDict[str, Any]
    q2_params: core.FrozenDict[str, Any]
    q1_target_params: core.FrozenDict[str, Any]
    q2_target_params: core.FrozenDict[str, Any]
    actor_params: core.FrozenDict[str, Any]
    log_entropy_coefficient: float
    entropy_coefficient: float
    qf1_optimizer_state: optax.OptState
    qf2_optimizer_state: optax.OptState
    actor_optimizer_state: optax.OptState
    entropy_optimizer_state: optax.OptState
    buffer_state: Any


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


def make_sac(args):

    # env, env_params = gymnax.make(args.env_id)
    env, env_params = BraxGymnaxWrapper(args.env_id), None
    env = LogWrapper(env)

    q_network = SoftQNetwork()
    actor = Actor(
        action_dim=jnp.array(env.action_space(env_params).shape).prod(),
        action_scale=(
            env.action_space(env_params).high - env.action_space(env_params).low
        )
        / 2,
        action_bias=(
            env.action_space(env_params).high + env.action_space(env_params).low
        )
        / 2,
    )
    buffer = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_batch_size=args.num_envs,
    )
    optimizer = optax.chain(optax.adam(learning_rate=args.learning_rate))

    target_entropy = -jnp.array(env.action_space(env_params).shape).prod()

    def init(key):
        key, reset_key, sample_key, step_key, q1_key, q2_key, actor_key = (
            jax.random.split(key, 7)
        )
        reset_keys = jax.random.split(reset_key, args.num_envs)
        sample_keys = jax.random.split(sample_key, args.num_envs)
        step_keys = jax.random.split(step_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        action = jax.vmap(env.action_space(env_params).sample)(sample_keys)
        _, _, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, env_state, action, env_params
        )

        q1_params = q_network.init(
            q1_key,
            obs,
            action,
        )
        q1_target_params = q_network.init(
            q1_key,
            obs,
            action,
        )
        q2_params = q_network.init(
            q2_key,
            obs,
            action,
        )
        q2_target_params = q_network.init(
            q2_key,
            obs,
            action,
        )
        actor_params = actor.init(
            actor_key,
            obs,
        )
        log_entropy_coefficient = jnp.array(0.0)
        entropy_coefficient = jnp.exp(log_entropy_coefficient)

        qf1_optimizer_state = optimizer.init(q1_params)
        qf2_optimizer_state = optimizer.init(q2_params)
        actor_optimizer_state = optimizer.init(actor_params)
        entropy_optimizer_state = optimizer.init(log_entropy_coefficient)

        transition = Transition(
            obs=obs[0],  # type: ignore
            action=action[0],  # type: ignore
            reward=reward[0],  # type: ignore
            done=done[0],  # type: ignore
        )  # type: ignore
        buffer_state = buffer.init(transition)

        return key, SACState(
            step=0,  # type: ignore
            obs=obs,  # type: ignore
            env_state=env_state,  # type: ignore
            q1_params=q1_params,  # type: ignore
            q2_params=q2_params,  # type: ignore
            q1_target_params=q1_target_params,  # type: ignore
            q2_target_params=q2_target_params,  # type: ignore
            actor_params=actor_params,  # type: ignore
            log_entropy_coefficient=log_entropy_coefficient,  # type: ignore
            entropy_coefficient=entropy_coefficient,  # type: ignore
            qf1_optimizer_state=qf1_optimizer_state,  # type: ignore
            qf2_optimizer_state=qf2_optimizer_state,  # type: ignore
            actor_optimizer_state=actor_optimizer_state,  # type: ignore
            entropy_optimizer_state=entropy_optimizer_state,  # type: ignore
            buffer_state=buffer_state,  # type: ignore
        )

    @partial(jax.jit, static_argnums=(2))
    def train(
        key,
        state: SACState,
        num_steps,
    ):

        def step(key, state):

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, args.num_envs)
            random_action = jax.vmap(env.action_space(env_params).sample)(sample_key)

            probs = actor.apply(
                state.actor_params,
                state.obs,
            )
            action, _ = actor.get_action(action_key, probs)

            action = jnp.where(
                state.step < args.learning_starts,
                random_action,
                action,
            )

            step_key = jax.random.split(step_key, args.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, env_params)

            transition = Transition(
                obs=state.obs,
                action=action,
                reward=reward,
                done=done,
            )

            state = state.replace(
                step=state.step + args.num_envs,
                obs=next_obs,
                env_state=env_state,
                buffer_state=buffer.add(state.buffer_state, transition),
            )

            return (
                key,
                state,
            ), info

        def update_critic(key, state, batch):

            probs = actor.apply(state.actor_params, batch.second.obs)
            next_action, next_log_prob = actor.get_action(key, probs)

            q1_next_target = q_network.apply(
                state.q1_target_params,
                batch.second.obs,
                next_action,
            ).squeeze(-1)
            q2_next_target = q_network.apply(
                state.q2_target_params,
                batch.second.obs,
                next_action,
            ).squeeze(-1)
            q_next_target = (
                jnp.minimum(q1_next_target, q2_next_target)
                - state.entropy_coefficient * next_log_prob
            )
            next_q_value = (
                batch.first.reward + (1 - batch.first.done) * args.gamma * q_next_target
            )

            def loss_fn(params):
                q_value = q_network.apply(
                    params,
                    batch.first.obs,
                    batch.first.action,
                ).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss

            (q1_loss), q1_grads = jax.value_and_grad(loss_fn)(state.q1_params)
            q1_updates, qf1_optimizer_state = optimizer.update(
                q1_grads, state.qf1_optimizer_state, state.q1_params
            )
            q1_params = optax.apply_updates(state.q1_params, q1_updates)

            (q2_loss), q2_grads = jax.value_and_grad(loss_fn)(state.q2_params)
            q2_updates, qf2_optimizer_state = optimizer.update(
                q2_grads, state.qf2_optimizer_state, state.q2_params
            )
            q2_params = optax.apply_updates(state.q2_params, q2_updates)

            q1_target_params = optax.periodic_update(
                state.q1_target_params,
                optax.incremental_update(q1_params, state.q1_target_params, args.tau),
                state.step.squeeze(),
                args.target_network_frequency,
            )
            q2_target_params = optax.periodic_update(
                state.q2_target_params,
                optax.incremental_update(q2_params, state.q2_target_params, args.tau),
                state.step.squeeze(),
                args.target_network_frequency,
            )
            state = state.replace(
                q1_params=q1_params,
                q2_params=q2_params,
                q1_target_params=q1_target_params,
                q2_target_params=q2_target_params,
                qf1_optimizer_state=qf1_optimizer_state,
                qf2_optimizer_state=qf2_optimizer_state,
            )

            return state, q1_loss + q2_loss

        def update_actor(key, state, batch):

            def loss_fn(params):

                probs = actor.apply(
                    params,
                    batch.first.obs,
                )
                action, log_prob = actor.get_action(key, probs)

                q1_value = q_network.apply(
                    state.q1_params,
                    batch.first.obs,
                    action,
                ).squeeze(-1)
                q2_value = q_network.apply(
                    state.q2_params,
                    batch.first.obs,
                    action,
                ).squeeze(-1)
                q_value = jnp.minimum(q1_value, q2_value)

                loss = ((state.entropy_coefficient * log_prob) - q_value).mean()
                return loss

            (loss), grads = jax.value_and_grad(loss_fn)(state.actor_params)
            updates, optimizer_state = optimizer.update(
                grads, state.actor_optimizer_state, state.actor_params
            )
            actor_params = optax.apply_updates(state.actor_params, updates)

            state = state.replace(
                actor_params=actor_params,
                actor_optimizer_state=optimizer_state,
            )

            return state, loss

        def update_entropy(key, state, batch):
            probs = actor.apply(
                state.actor_params,
                batch.first.obs,
            )
            _, log_prob = actor.get_action(key, probs)

            def loss_fn(params):
                loss = (-jnp.exp(params) * (log_prob + target_entropy)).mean()
                return loss

            (loss), grads = jax.value_and_grad(loss_fn)(state.log_entropy_coefficient)
            updates, optimizer_state = optimizer.update(
                grads, state.entropy_optimizer_state, state.log_entropy_coefficient
            )
            log_entropy_coefficient = optax.apply_updates(
                state.log_entropy_coefficient, updates
            )
            entropy_coefficient = jnp.exp(log_entropy_coefficient)

            state = state.replace(
                log_entropy_coefficient=log_entropy_coefficient,
                entropy_coefficient=entropy_coefficient,
                entropy_optimizer_state=optimizer_state,
            )
            return state, loss

        def update_step(carry, _):

            key, state = carry

            (key, state), info = step(key, state)

            key, buffer_key, critic_key, actor_key, entropy_key = jax.random.split(
                key, 5
            )

            batch = buffer.sample(state.buffer_state, buffer_key).experience
            state, loss = jax.lax.cond(
                state.step.squeeze() < args.learning_starts,
                lambda *_: (state, jnp.array(0.0)),
                update_critic,
                critic_key,
                state,
                batch,
            )

            state, loss = jax.lax.cond(
                (state.step.squeeze() < args.learning_starts)
                | (state.step.squeeze() % args.policy_frequency != 0),
                lambda *_: (state, jnp.array(0.0)),
                update_actor,
                actor_key,
                state,
                batch,
            )
            state, loss = jax.lax.cond(
                (state.step.squeeze() < args.learning_starts),
                lambda *_: (state, jnp.array(0.0)),
                update_entropy,
                entropy_key,
                state,
                batch,
            )

            return (
                key,
                state,
            ), info

        (
            key,
            state,
        ), info = jax.lax.scan(
            update_step,
            (
                key,
                state,
            ),
            length=(num_steps),
        )
        return (
            key,
            state,
            info,
        )

    @partial(jax.jit, static_argnums=(2))
    def evaluate(key, state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

        def step(carry, _):
            key, obs, env_state = carry

            key, action_key, step_key = jax.random.split(key, 3)

            probs = actor.apply(
                state.actor_params,
                obs,
            )
            action, _ = actor.get_action(action_key, probs)

            step_key = jax.random.split(step_key, args.num_envs)
            obs, env_state, _, _, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, env_state, action, env_params
            )

            return (key, obs, env_state), info

        (key, *_), info = jax.lax.scan(step, (key, obs, env_state), length=num_steps)

        return key, info

    return SAC(
        init=init,
        train=train,
        evaluate=evaluate,
    )
