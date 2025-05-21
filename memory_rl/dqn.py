import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
from flax import core
import gymnax
import jax
import jax.numpy as jnp
import optax

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
    buffer_size: int = 100_000
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
    """
    Convolutional Q-network mapping observations to action Q-values.
    """

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
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_dim)(x)

        return q_values


@chex.dataclass(frozen=True)
class DQNState:
    """
    Immutable container for training state of DQN algorithm.
    """

    step: int
    obs: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState


@chex.dataclass(frozen=True)
class DQN:
    """
    Deep Q-Network (DQN) reinforcement learning algorithm.
    """

    args: Args
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: QNetwork
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule

    def create(self, key) -> tuple[chex.PRNGKey, DQNState, chex.Array, gymnax.EnvState]:
        """
        Initialize environment, network parameters, optimizer, and replay buffer.

        Args:
            key: JAX PRNG key for randomness.

        Returns:
            key: Updated PRNG key after splits.
            state: Initialized DQNState with params, target_params, optimizer_state, buffer_state.
            obs: Initial observations from vectorized envs.
            env_state: Initial environment state.
        """
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.args.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        params = self.q_network.init(q_key, obs)
        target_params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        transition = Transition(obs=obs[0], action=action[0], reward=reward[0], done=done[0])  # type: ignore
        buffer_state = self.buffer.init(transition)

        return (
            key,
            DQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                target_params=target_params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                buffer_state=buffer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: chex.PRNGKey, state: DQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, DQNState]:

        def step(carry, _):

            key, state = carry

            key, sample_key, step_key = jax.random.split(key, 3)

            sample_key = jax.random.split(sample_key, self.args.num_envs)
            action = jax.vmap(self.env.action_space(self.env_params).sample)(sample_key)

            step_key = jax.random.split(step_key, self.args.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=state.obs,  # type: ignore
                action=action,  # type: ignore
                reward=reward,  # type: ignore
                done=done,  # type: ignore
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                step=state.step + self.args.num_envs,
                obs=next_obs,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            if self.args.debug:

                def callback(info):
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

                jax.debug.callback(callback, info)

            return (key, state), info

        (key, state), _ = jax.lax.scan(
            step, (key, state), length=num_steps // self.args.num_envs
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: chex.PRNGKey,
        state: DQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, DQNState, dict]:
        """
        Run training loop for specified number of steps.

        Args:
            key: JAX PRNG key.
            state: Current DQNState.
            obs: Current env observations.
            env_state: Current env state.
            num_steps: Total environment steps to train.

        Returns:
            key: Updated PRNG key.
            state: Updated DQNState after training.
            obs: Latest observations.
            env_state: Latest env state.
            info: Training statistics (loss, rewards, etc.).
        """

        def step(carry, _):

            key, state = carry

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, self.args.num_envs)
            random_action = jax.vmap(self.env.action_space(self.env_params).sample)(
                sample_key
            )

            q_values = self.q_network.apply(state.params, state.obs)
            greedy_action = q_values.argmax(axis=-1)

            epsilon = self.epsilon_schedule(state.step)
            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape) < epsilon,
                random_action,
                greedy_action,
            ).squeeze()

            step_key = jax.random.split(step_key, self.args.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=state.obs,  # type: ignore
                action=action,  # type: ignore
                reward=reward,  # type: ignore
                done=done,  # type: ignore
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                step=state.step + self.args.num_envs,
                obs=next_obs,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (key, state), info

        def update(key: chex.PRNGKey, state: DQNState) -> tuple[DQNState, chex.Array]:

            batch = self.buffer.sample(state.buffer_state, key).experience

            q_next_target = self.q_network.apply(
                state.target_params,
                batch.second.obs,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.first.reward
                + (1 - batch.first.done) * self.args.gamma * q_next_target
            )

            def loss_fn(params):
                q_value = self.q_network.apply(
                    params,
                    batch.first.obs,
                )
                action = jnp.expand_dims(batch.first.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss

            (loss), grads = jax.value_and_grad(loss_fn)(state.params)
            updates, optimizer_state = self.optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = optax.periodic_update(
                optax.incremental_update(params, state.target_params, self.args.tau),
                state.target_params,
                state.step,
                self.args.target_network_frequency,
            )

            state = state.replace(
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
            )

            return state, loss

        def learn(
            carry: tuple[chex.PRNGKey, DQNState], _
        ) -> tuple[tuple[chex.PRNGKey, DQNState], dict]:

            (key, state), info = jax.lax.scan(
                step, carry, length=self.args.train_frequency // self.args.num_envs
            )

            key, update_key = jax.random.split(key)
            state, loss = update(update_key, state)

            if self.args.debug:

                def callback(info):
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

                jax.debug.callback(callback, info)

            return (key, state), info

        (key, state), info = jax.lax.scan(
            learn,
            (key, state),
            length=(num_steps // self.args.train_frequency),
        )
        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(
        self, key: chex.PRNGKey, state: DQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:
        """
        Evaluate current policy for a fixed number of steps without exploration.

        Args:
            key: JAX PRNG key.
            state: DQNState with trained parameters.
            num_steps: Number of evaluation steps.

        Returns:
            key: Updated PRNG key.
            info: Evaluation metrics (rewards, episode lengths, etc.).
        """
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        def step(carry: tuple[chex.PRNGKey, chex.Array, gymnax.EnvState], _):
            key, obs, env_state = carry

            q_values = self.q_network.apply(
                state.params,
                obs,
            )
            action = q_values.argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.args.num_envs)
            obs, env_state, _, _, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, env_state), info

        (key, *_), info = jax.lax.scan(step, (key, obs, env_state), length=num_steps)

        return key, info


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


def make_dqn(args: Args) -> DQN:
    """
    Factory function to construct a DQN agent from Args.

    Args:
        args: Experiment configuration.

    Returns:
        An initialized DQN instance ready for training.
    """
    env, env_params = gymnax.make(args.env_id)
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
    return DQN(
        args=args,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
