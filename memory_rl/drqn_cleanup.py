import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import flax.linen as nn
import gymnax
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from flax import core
from gymnax.wrappers.purerl import FlattenObservationWrapper
from popjaxrl.envs import make

from networks import MaskedOptimizedLSTMCell, MaskedRNN
from utils import LogWrapper, make_trajectory_buffer, periodic_incremental_update


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
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    cell_size: int = 128
    """the size of the cell in the LSTM"""
    num_envs: int = 10
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    sample_sequence_length: int = 4
    """the length of the sequence to sample from the replay memory"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    update_hidden_state: bool = True
    """whether to update the hidden state of the network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    num_epoch_steps: int = 100_000
    """the number of steps for each epoch"""
    num_evaluation_steps: int = 1_000
    """the number of steps for each evaluation"""

    @property
    def num_updates(self):
        return self.total_timesteps // self.num_envs

    @property
    def num_epochs(self):
        return self.total_timesteps // self.num_epoch_steps


class QNetwork(nn.Module):
    action_dim: int
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jax.Array | None = None,
        return_carry_history: bool = False,
    ):
        # TODO: Figure out how to use CNNs with RNNs
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        h, x = MaskedRNN(self.cell, return_carry=True)(  # type: ignore
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        q_values = nn.Dense(self.action_dim)(x)

        return h, q_values

    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.cell.initialize_carry(key, input_shape)


@chex.dataclass(frozen=True)
class DRQNState:
    step: int
    obs: chex.Array
    done: chex.Array
    hidden_state: tuple
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: Any


@chex.dataclass(frozen=True)
class DRQN:
    args: Args
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: QNetwork
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule

    @partial(jax.jit, static_argnames=["self"])
    def create(self, key):
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.args.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        c, h = self.q_network.initialize_carry(
            (self.args.num_envs, self.args.cell_size)
        )

        params = self.q_network.init(
            q_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            (c, h),
        )
        target_params = self.q_network.init(
            q_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            (c, h),
        )
        optimizer_state = self.optimizer.init(params)

        transition = Transition(obs=obs[0], done=done[0], hidden_state=(c[0], h[0]), action=action[0], reward=reward[0], next_obs=obs[0], next_done=done[0], next_hidden_state=(c[0], h[0]))  # type: ignore
        buffer_state = self.buffer.init(transition)

        return (
            key,
            DRQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                hidden_state=(c, h),  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                target_params=target_params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                buffer_state=buffer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: chex.PRNGKey,
        state: DRQNState,
        num_steps: int,
    ):

        def step(carry: tuple, _):

            (
                key,
                state,
            ) = carry

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, self.args.num_envs)
            random_action = jax.vmap(self.env.action_space(self.env_params).sample)(
                sample_key
            )

            next_hidden_state, q_values = self.q_network.apply(
                state.params,
                jnp.expand_dims(state.obs, 1),
                jnp.expand_dims(state.done, 1),
                state.hidden_state,
            )
            greedy_action = q_values.squeeze(1).argmax(axis=-1)

            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape)
                < self.epsilon_schedule(state.step),
                random_action,
                greedy_action,
            ).squeeze()

            step_key = jax.random.split(step_key, self.args.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=jnp.expand_dims(state.obs, 1),
                done=jnp.expand_dims(state.done, 1),
                hidden_state=jax.tree.map(
                    lambda x: jnp.expand_dims(x, 1),
                    state.hidden_state,
                ),
                action=jnp.expand_dims(action, 1),
                reward=jnp.expand_dims(reward, 1),
                next_obs=jnp.expand_dims(next_obs, 1),
                next_done=jnp.expand_dims(next_done, 1),
                next_hidden_state=jax.tree.map(
                    lambda x: jnp.expand_dims(x, 1),
                    next_hidden_state,
                ),
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                step=state.step + self.args.num_envs,
                obs=next_obs,  # type: ignore
                done=next_done,  # type: ignore
                hidden_state=next_hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (
                key,
                state,
            ), info

        def update(key: chex.PRNGKey, state: DRQNState) -> tuple[DRQNState, chex.Array]:

            batch = self.buffer.sample(state.buffer_state, key)

            next_hidden_state, q_next_target = self.q_network.apply(
                state.target_params,
                batch.experience.next_obs,
                batch.experience.next_done,
                jax.tree_map(lambda x: x[:, 0, :], batch.experience.next_hidden_state),
                return_carry_history=self.args.update_hidden_state,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.experience.reward
                + (1 - batch.experience.next_done) * self.args.gamma * q_next_target
            )

            def loss_fn(params):
                hidden_state, q_value = self.q_network.apply(
                    params,
                    batch.experience.obs,
                    batch.experience.done,
                    jax.tree_map(lambda x: x[:, 0, :], batch.experience.hidden_state),
                    return_carry_history=self.args.update_hidden_state,
                )
                action = jnp.expand_dims(batch.experience.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss, hidden_state

            (loss, hidden_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            updates, optimizer_state = self.optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = periodic_incremental_update(
                params,
                state.target_params,
                state.step,
                self.args.target_network_frequency,
                self.args.tau,
            )

            if self.args.update_hidden_state:
                hidden_state = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1), hidden_state
                )
                next_hidden_state = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1), next_hidden_state
                )
                experience = batch.experience.replace(
                    hidden_state=hidden_state,
                    next_hidden_state=next_hidden_state,
                )
                state = state.replace(
                    buffer_state=self.buffer.update(
                        state.buffer_state,
                        experience,
                        batch.sampled_batch_indices,
                        batch.traj_time_indices,
                    )
                )

            state = state.replace(
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
            )

            return state, loss

        def update_step(carry, _):

            (key, state), info = jax.lax.scan(
                step, carry, length=self.args.train_frequency // self.args.num_envs
            )

            key, update_key = jax.random.split(key)

            state, loss = jax.lax.cond(
                state.step.squeeze() < self.args.learning_starts,
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
            length=(num_steps // self.args.train_frequency),
        )
        return (
            key,
            state,
            info,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(
        self, key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jax.vmap(self.env.is_terminal, in_axes=(0, None))(
            env_state.env_state, self.env_params
        )
        hidden_state = self.q_network.initialize_carry(
            (self.args.num_envs, self.args.cell_size)
        )

        def step(
            carry: tuple[chex.PRNGKey, chex.Array, chex.Array, tuple, gymnax.EnvState],
            _,
        ):
            key, obs, done, hidden_state, env_state = carry

            hidden_state, q_values = self.q_network.apply(
                state.params,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                hidden_state,
            )
            action = q_values.squeeze(1).argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.args.num_envs)
            obs, env_state, _, _, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, done, hidden_state, env_state), info

        (key, *_), info = jax.lax.scan(
            step,
            (key, obs, done, hidden_state, env_state),
            length=num_steps // self.args.num_envs,
        )

        return key, info


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    done: chex.Array
    hidden_state: tuple
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    next_done: chex.Array
    next_hidden_state: tuple


def make_drqn(args: Args) -> DRQN:

    env, env_params = gymnax.make(args.env_id)
    # env, env_params = make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    q_network = QNetwork(
        action_dim=env.action_space(env_params).n,
        cell=MaskedOptimizedLSTMCell(args.cell_size),
    )
    buffer = make_trajectory_buffer(
        add_batch_size=args.num_envs,
        sample_batch_size=args.batch_size,
        sample_sequence_length=args.sample_sequence_length,
        period=1,
        min_length_time_axis=args.sample_sequence_length,
        max_size=args.buffer_size,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adam(learning_rate=args.learning_rate)
    )
    epsilon_schedule = optax.linear_schedule(
        args.start_e,
        args.end_e,
        int(args.exploration_fraction * args.total_timesteps),
        args.learning_starts,
    )

    return DRQN(
        args=args,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
