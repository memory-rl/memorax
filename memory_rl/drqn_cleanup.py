import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from gymnax.wrappers.purerl import FlattenObservationWrapper

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
    num_evaluation_steps: int = 10_000
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
    env_state: Any
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: Any


@chex.dataclass(frozen=True)
class DRQN:
    init: Callable[[chex.PRNGKey], DRQNState]
    train: Callable[
        [chex.PRNGKey, DRQNState, int], tuple[chex.PRNGKey, DRQNState, dict]
    ]
    evaluate: Callable[[chex.PRNGKey, DRQNState, int], tuple[chex.PRNGKey, dict]]


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

    def init(key: chex.PRNGKey) -> tuple[chex.PRNGKey, DRQNState]:
        key, reset_key, sample_key, step_key, q_key = jax.random.split(key, 5)
        reset_keys = jax.random.split(reset_key, args.num_envs)
        sample_keys = jax.random.split(sample_key, args.num_envs)
        step_keys = jax.random.split(step_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        action = jax.vmap(env.action_space(env_params).sample)(sample_keys)
        _, _, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, env_state, action, env_params
        )
        c, h = q_network.initialize_carry((args.num_envs, args.cell_size))

        params = q_network.init(
            q_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            (c, h),
        )
        target_params = q_network.init(
            q_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            (c, h),
        )
        optimizer_state = optimizer.init(params)

        transition = Transition(obs=obs[0], done=done[0], hidden_state=(c[0], h[0]), action=action[0], reward=reward[0], next_obs=obs[0], next_done=done[0], next_hidden_state=(c[0], h[0]))  # type: ignore
        buffer_state = buffer.init(transition)

        return key, DRQNState(
            step=0,
            obs=obs,
            done=done,
            hidden_state=(c, h),
            env_state=env_state,
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

    @partial(jax.jit, static_argnums=(2))
    def train(
        key: chex.PRNGKey,
        state: DRQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, DRQNState, dict]:

        def step(
            carry: tuple[chex.PRNGKey, DRQNState], _
        ) -> tuple[tuple[chex.PRNGKey, DRQNState], dict]:

            (
                key,
                state,
            ) = carry

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, args.num_envs)
            random_action = jax.vmap(env.action_space(env_params).sample)(sample_key)

            hidden_state, q_values = q_network.apply(
                state.params,
                jnp.expand_dims(state.obs, 1),
                jnp.expand_dims(state.done, 1),
                state.hidden_state,
            )
            greedy_action = q_values.squeeze(1).argmax(axis=-1)

            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape)
                < epsilon_schedule(state.step),
                random_action,
                greedy_action,
            ).squeeze()

            step_key = jax.random.split(step_key, args.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, env_params)

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
                    hidden_state,
                ),
            )

            state = state.replace(
                step=state.step + args.num_envs,
                obs=next_obs,
                done=next_done,
                hidden_state=hidden_state,
                env_state=env_state,
                buffer_state=buffer.add(state.buffer_state, transition),
            )

            return (
                key,
                state,
            ), info

        def update(key: chex.PRNGKey, state: DRQNState) -> tuple[DRQNState, chex.Array]:

            batch = buffer.sample(state.buffer_state, key)

            next_hidden_state, q_next_target = q_network.apply(
                state.target_params,
                batch.experience.next_obs,
                batch.experience.next_done,
                jax.tree_map(lambda x: x[:, 0, :], batch.experience.next_hidden_state),
                return_carry_history=args.update_hidden_state,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.experience.reward
                + (1 - batch.experience.next_done) * args.gamma * q_next_target
            )

            def loss_fn(params):
                hidden_state, q_value = q_network.apply(
                    params,
                    batch.experience.obs,
                    batch.experience.done,
                    jax.tree_map(lambda x: x[:, 0, :], batch.experience.hidden_state),
                    return_carry_history=args.update_hidden_state,
                )
                action = jnp.expand_dims(batch.experience.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss, hidden_state

            (loss, hidden_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            updates, optimizer_state = optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = periodic_incremental_update(
                params,
                state.target_params,
                state.step.squeeze(),
                args.target_network_frequency,
                args.tau,
            )

            if args.update_hidden_state:
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
                    buffer_state=buffer.update(
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

        def update_step(
            carry: tuple[chex.PRNGKey, DRQNState], _
        ) -> tuple[tuple[chex.PRNGKey, DRQNState], dict]:

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
    def evaluate(
        key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(
            env_state.env_state, env_params
        )
        hidden_state = q_network.initialize_carry((args.num_envs, args.cell_size))

        def step(
            carry: tuple[chex.PRNGKey, chex.Array, chex.Array, tuple, gymnax.EnvState],
            _,
        ):
            key, obs, done, hidden_state, env_state = carry

            hidden_state, q_values = q_network.apply(
                state.params,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                hidden_state,
            )
            action = q_values.squeeze(1).argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, args.num_envs)
            obs, env_state, _, _, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, env_state, action, env_params
            )

            return (key, obs, done, hidden_state, env_state), info

        (key, *_), info = jax.lax.scan(
            step,
            (key, obs, done, hidden_state, env_state),
            length=num_steps // args.num_envs,
        )

        return key, info

    return DRQN(
        init=init,
        train=train,
        evaluate=evaluate,
    )
