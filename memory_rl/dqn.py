import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from gymnax.wrappers.purerl import FlattenObservationWrapper

import wandb
from utils import LogWrapper


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
    env_id: str = "Breakout-MinAtar"
    """the id of the environment"""
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    buffer_size: int = 10_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
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


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ):
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
        )(x)
        # x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_dim)(x)

        return q_values


@chex.dataclass(frozen=True)
class DQN:
    init: Callable
    train: Callable
    evaluate: Callable


@chex.dataclass(frozen=True)
class DQNState:
    step: int
    obs: chex.Array
    env_state: Any
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: Any


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


def make_env_state(key, env, env_params, num_envs):
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    return key, obs, env_state


def make_buffer_state(key, buffer, env, env_params):
    obs, env_state = env.reset(key, env_params)
    action = env.action_space(env_params).sample(key)
    _, env_state, reward, done, _ = env.step(key, env_state, action, env_params)
    transition = Transition(obs=obs, action=action, reward=reward, done=done)  # type: ignore
    buffer_state = buffer.init(transition)
    return key, buffer_state


def make_train_state(key, q_network, optimizer, env, env_params, num_envs):
    key, reset_key, q_key = jax.random.split(key, 3)

    keys = jax.random.split(reset_key, num_envs)
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

    params = q_network.init(
        q_key,
        obs,
    )
    target_params = q_network.init(
        q_key,
        obs,
    )
    optimizer_state = optimizer.init(params)
    return key, params, target_params, optimizer_state


def make_dqn(args):

    env, env_params = gymnax.make(args.env_id)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    q_network = QNetwork(
        action_dim=env.action_space(env_params).n,
    )
    buffer = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_sequences=False,
        add_batch_size=args.num_envs,
    )
    optimizer = optax.adam(learning_rate=args.learning_rate)
    epsilon_schedule = optax.linear_schedule(
        args.start_e,
        args.end_e,
        int(args.exploration_fraction * args.total_timesteps),
        args.learning_starts,
    )

    def init(key):
        key, obs, env_state = make_env_state(key, env, env_params, args.num_envs)
        key, params, target_params, optimizer_state = make_train_state(
            key, q_network, optimizer, env, env_params, args.num_envs
        )
        key, buffer_state = make_buffer_state(key, buffer, env, env_params)
        return key, DQNState(
            step=0,
            obs=obs,
            env_state=env_state,
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

    @partial(jax.jit, static_argnums=(2))
    def train(
        key,
        state: DQNState,
        num_steps,
    ):

        def step(carry, _):

            (
                key,
                state,
            ) = carry

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, args.num_envs)
            random_action = jax.vmap(env.action_space(env_params).sample)(sample_key)

            q_values = q_network.apply(
                state.params,
                state.obs,
            )
            greedy_action = q_values.argmax(axis=-1)

            epsilon = epsilon_schedule(state.step)
            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape) < epsilon,
                random_action,
                greedy_action,
            ).squeeze()

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

        def update(key, state):

            batch = buffer.sample(state.buffer_state, key).experience

            q_next_target = q_network.apply(
                state.target_params,
                batch.second.obs,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.first.reward + (1 - batch.first.done) * args.gamma * q_next_target
            )

            def loss_fn(params):
                q_value = q_network.apply(
                    params,
                    batch.first.obs,
                )
                action = jnp.expand_dims(batch.first.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss

            (loss), grads = jax.value_and_grad(loss_fn)(state.params)
            updates, optimizer_state = optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = optax.periodic_update(
                state.target_params,
                optax.incremental_update(params, state.target_params, args.tau),
                state.step.squeeze(),
                args.target_network_frequency,
            )

            state = state.replace(
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
            )

            return state, loss

        def update_step(carry, _):

            (key, state), info = jax.lax.scan(
                step, carry, length=args.train_frequency // args.num_envs
            )

            key, update_key = jax.random.split(key)

            state, loss = jax.lax.cond(
                state.step.squeeze() < args.learning_starts,
                lambda *_: (state, jnp.array(0.0)),
                update,
                update_key,
                state,
            )
            info["loss"] = loss

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
            length=(num_steps // args.train_frequency),
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

            q_values = q_network.apply(
                state.params,
                obs,
            )
            action = q_values.argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, args.num_envs)
            obs, env_state, _, _, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, env_state, action, env_params
            )

            return (key, obs, env_state), info

        (key, *_), info = jax.lax.scan(step, (key, obs, env_state), length=num_steps)

        return key, info

    return DQN(
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
        init=init,
        train=train,
        evaluate=evaluate,
    )
