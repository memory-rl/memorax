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
import tyro
from flax import core, struct
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper

import wandb
from networks import MaskedLSTMCell, MaskedOptimizedLSTMCell, MaskedRNN
from utils import LogWrapper, make_trajectory_buffer


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
    debug: bool = False
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
    num_envs: int = 10
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    sample_sequence_length: int = 4
    """the sequence length for TBPTT"""
    burn_in_length: int = 4
    """the length of burn-in"""
    update_hidden_states: bool = True
    """whether to update the hidden states of the recurrent network"""
    initalize_hidden_state: bool = True
    """whether to initialize the hidden states of the recurrent network in the update"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
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
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jax.Array | None = None,
        return_carry_history: bool = False,
        burn_in_length: int = 0,
    ):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        (c, h), x = MaskedRNN(self.cell, return_carry=True)(  # type: ignore
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
            burn_in_length=burn_in_length,
        )
        q_values = nn.Dense(self.action_dim)(x)

        return (c, h), q_values


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


class TrainState(TrainState):
    target_params: core.FrozenDict
    hidden_state: tuple
    initialize_carry: Callable = struct.field(pytree_node=False)
    timesteps: int
    n_updates: int


def make_train(args):

    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    q_network = QNetwork(
        action_dim=env.action_space(env_params).n,
        cell=MaskedOptimizedLSTMCell(
            128,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0),
        ),
    )

    sample_sequence_length: int = args.sample_sequence_length + args.burn_in_length
    buffer = make_trajectory_buffer(
        add_batch_size=args.num_envs,
        sample_batch_size=args.batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        min_length_time_axis=sample_sequence_length,
        max_size=args.buffer_size,
    )

    def make_train_state(key):
        key, reset_key, q_key = jax.random.split(key, 3)

        keys = jax.random.split(reset_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)
        action = jax.vmap(env.action_space(env_params).sample)(keys)
        *_, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            keys, env_state, action, env_params
        )
        hidden_state = q_network.cell.initialize_carry(key, (args.num_envs, 128))

        q_state = TrainState.create(
            apply_fn=q_network.apply,
            params=q_network.init(
                q_key,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                hidden_state,
            ),
            target_params=q_network.init(
                q_key,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                hidden_state,
            ),
            hidden_state=hidden_state,
            tx=optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(
                    learning_rate=args.learning_rate,
                ),
            ),
            initialize_carry=q_network.cell.initialize_carry,
            timesteps=0,
            n_updates=0,
        )

        obs, env_state = env.reset(key, env_params)
        action = env.action_space(env_params).sample(key)
        _, env_state, reward, done, _ = env.step(key, env_state, action, env_params)
        hidden_state = (jnp.zeros(128), jnp.zeros(128))
        transition = Transition(obs=obs, done=done, hidden_state=hidden_state, action=action, reward=reward, next_obs=obs, next_done=done, next_hidden_state=hidden_state)  # type: ignore
        buffer_state = buffer.init(transition)

        return key, env_state, q_state, buffer_state

    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return jnp.maximum(slope * t + start_e, end_e)

    @partial(jax.jit, static_argnums=(4))
    def train(key, env_state, q_state, buffer_state, num_steps):

        key, reset_key = jax.random.split(key)

        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(
            env_state.env_state, env_params
        )
        q_state = q_state.replace(
            hidden_state=q_network.cell.initialize_carry(key, (args.num_envs, 128))
        )

        def update_step(carry, _):

            key, q_state, buffer_state, env_state, obs, done = carry

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                q_state.timesteps,
            )

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, args.num_envs)
            random_action = jax.vmap(env.action_space(env_params).sample)(sample_key)

            hidden_state, q_values = q_state.apply_fn(
                q_state.params,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                q_state.hidden_state,
            )
            greedy_action = q_values.squeeze(1).argmax(axis=-1)

            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape) < epsilon,
                random_action,
                greedy_action,
            )

            step_key = jax.random.split(step_key, args.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, env_params)

            transition = Transition(
                obs=jnp.expand_dims(obs, 1),
                done=jnp.expand_dims(done, 1),
                hidden_state=jax.tree_map(
                    lambda x: jnp.expand_dims(x, 1), q_state.hidden_state
                ),
                action=jnp.expand_dims(action, 1),
                reward=jnp.expand_dims(reward, 1),
                next_obs=jnp.expand_dims(next_obs, 1),
                next_done=jnp.expand_dims(next_done, 1),
                next_hidden_state=jax.tree_map(
                    lambda x: jnp.expand_dims(x, 1), hidden_state
                ),
            )

            q_state = q_state.replace(
                timesteps=q_state.timesteps + args.num_envs, hidden_state=hidden_state
            )
            buffer_state = buffer.add(buffer_state, transition)

            def update(key, q_state, buffer_state):

                sample = buffer.sample(buffer_state, key)
                batch = sample.experience

                initial_carry = jax.tree_map(
                    lambda x: x[:, 0, :], batch.next_hidden_state
                )

                next_hidden_states, q_next_target = q_state.apply_fn(
                    q_state.target_params,
                    batch.next_obs,
                    batch.next_done,
                    initial_carry=initial_carry,
                    return_carry_history=args.update_hidden_states,
                    burn_in_length=args.burn_in_length,
                )
                q_next_target = jnp.max(q_next_target, axis=-1)

                reward = batch.reward[:, args.burn_in_length :]
                next_done = batch.next_done[:, args.burn_in_length :]
                next_q_value = reward + (1 - next_done) * args.gamma * q_next_target

                def loss_fn(params):
                    initial_carry = jax.tree_map(
                        lambda x: x[:, 0, :], batch.hidden_state
                    )
                    hidden_states, q_value = q_state.apply_fn(
                        params,
                        batch.obs,
                        batch.done,
                        initial_carry=initial_carry,
                        return_carry_history=args.update_hidden_states,
                        burn_in_length=args.burn_in_length,
                    )
                    action = batch.action[:, args.burn_in_length :]
                    action = jnp.expand_dims(action, axis=-1)
                    q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                    loss = jnp.square(q_value - next_q_value).mean()
                    return loss, hidden_states

                (loss, hidden_states), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(q_state.params)
                q_state = q_state.apply_gradients(grads=grads)
                q_state = q_state.replace(n_updates=q_state.n_updates + 1)

                if args.update_hidden_states:

                    hidden_states = jax.tree.map(
                        lambda x: jnp.swapaxes(x, 0, 1), hidden_states
                    )
                    hidden_states = jax.tree.map(
                        lambda batch, hidden_states: batch.at[
                            :, args.burn_in_length :
                        ].set(hidden_states),
                        batch.hidden_state,
                        hidden_states,
                    )
                    next_hidden_states = jax.tree.map(
                        lambda x: jnp.swapaxes(x, 0, 1), next_hidden_states
                    )

                    next_hidden_states = jax.tree.map(
                        lambda batch, next_hidden_states: batch.at[
                            :, args.burn_in_length :
                        ].set(next_hidden_states),
                        batch.next_hidden_state,
                        next_hidden_states,
                    )

                    batch = batch.replace(
                        hidden_state=hidden_states,
                        next_hidden_state=next_hidden_states,
                    )
                    buffer_state = buffer.update(
                        buffer_state,
                        batch,
                        sample.sampled_batch_indices,
                        sample.traj_time_indices,
                    )

                return q_state, buffer_state, loss

            key, update_key = jax.random.split(key)
            q_state, buffer_state, loss = jax.lax.cond(
                (
                    buffer.can_sample(buffer_state)
                    & (q_state.timesteps > args.learning_starts)
                    & (q_state.timesteps % args.train_frequency == 0)
                ),
                lambda key, q_state: update(key, q_state, buffer_state),
                lambda _, q_state: (q_state, buffer_state, jnp.array(0.0)),
                update_key,
                q_state,
            )

            target_params = optax.incremental_update(
                q_state.params,
                q_state.target_params,
                args.tau,
            )
            q_state = jax.lax.cond(
                q_state.timesteps % args.target_network_frequency == 0,
                lambda q_state: q_state.replace(
                    target_params=target_params,
                ),
                lambda train_state: train_state,
                operand=q_state,
            )

            if args.debug:

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                        if args.track:
                            wandb.log(
                                {"episodic return": return_values[t]}, step=timesteps[t]
                            )

                jax.debug.callback(callback, info)

            return (
                key,
                q_state,
                buffer_state,
                env_state,
                next_obs,
                next_done,
            ), info

        (key, q_state, buffer_state, *_), info = jax.lax.scan(
            update_step,
            (key, q_state, buffer_state, env_state, obs, done),
            length=(num_steps // args.num_envs),
        )
        return key, env_state, q_state, buffer_state, info

    return train, make_train_state


def make_evaluate(args: Args):
    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    @partial(jax.jit, static_argnums=(2))
    def evaluate(key, agent_state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)
        initial_hidden_state = agent_state.initialize_carry(key, (args.num_envs, 128))

        def step(carry, _):
            key, obs, state, done, hidden_state = carry

            hidden_state, q_values = agent_state.apply_fn(
                agent_state.params,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                hidden_state,
            )
            action = q_values.squeeze(1).argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, args.num_envs)
            obs, state, _, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, state, action, env_params
            )

            return (key, obs, state, done, hidden_state), info

        (key, *_), info = jax.lax.scan(
            step, (key, obs, state, done, initial_hidden_state), length=num_steps
        )

        return key, info

    return evaluate


def main():
    args = tyro.cli(Args)

    assert (
        args.train_frequency % args.num_envs == 0
    ), f"train_frequency must be divisible by num_envs, but got {args.train_frequency} and {args.num_envs}"
    assert (
        args.target_network_frequency % args.num_envs == 0
    ), f"target_network_frequency must be divisible by num_envs, but got {args.target_network_frequency} and {args.num_envs}"

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)

    train, make_train_state = make_train(args)
    evaluate = make_evaluate(args)

    key = jax.random.key(args.seed)

    key, env_state, q_state, buffer_state = make_train_state(key)

    key, info = evaluate(key, q_state, args.num_evaluation_steps)
    print(f"Initial Return: {info['returned_episode_returns'].mean()}")

    for epoch in range(1, args.num_epochs):
        key, env_state, q_state, buffer_state, info = train(
            key, env_state, q_state, buffer_state, args.num_epoch_steps
        )
        key, info = evaluate(key, q_state, args.num_evaluation_steps)
        print(
            f"Timestep: {epoch*args.num_epoch_steps}, Return: {info['returned_episode_returns'].mean()}"
        )

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(info["returned_episode_returns"].mean(-1).reshape(-1))
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.show()


if __name__ == "__main__":
    main()
