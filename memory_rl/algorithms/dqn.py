from functools import partial
from typing import Any

import chex
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.loggers import Logger
from memory_rl.networks import Network, heads
from memory_rl.utils import Transition


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

    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[chex.PRNGKey, DQNState, chex.Array, gymnax.EnvState]:
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
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

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

            sample_key = jax.random.split(sample_key, self.cfg.algorithm.num_envs)
            action = jax.vmap(self.env.action_space(self.env_params).sample)(sample_key)

            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
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
                obs=next_obs,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (key, state), info

        (key, state), _ = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.algorithm.num_envs
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

            sample_key = jax.random.split(sample_key, self.cfg.algorithm.num_envs)
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
            )

            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
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
                step=state.step + self.cfg.algorithm.num_envs,
                obs=next_obs,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (key, state), info

        def update(
            key: chex.PRNGKey, state: DQNState
        ) -> tuple[DQNState, chex.Array, chex.Array]:

            key, sample_key = jax.random.split(key)
            batch = self.buffer.sample(state.buffer_state, sample_key).experience

            q_next_target = self.q_network.apply(
                state.target_params,
                batch.second.obs,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.first.reward
                + (1 - batch.first.done) * self.cfg.algorithm.gamma * q_next_target
            )

            def loss_fn(params):
                q_value = self.q_network.apply(
                    params,
                    batch.first.obs,
                )
                action = jnp.expand_dims(batch.first.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss, q_value

            (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            updates, optimizer_state = self.optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = optax.periodic_update(
                optax.incremental_update(
                    params, state.target_params, self.cfg.algorithm.tau
                ),
                state.target_params,
                state.step,
                self.cfg.algorithm.target_network_frequency,
            )

            state = state.replace(
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
            )

            return key, state, loss, q_value.mean()

        def learn(
            carry: tuple[chex.PRNGKey, DQNState], _
        ) -> tuple[tuple[chex.PRNGKey, DQNState], dict]:

            (key, state), info = jax.lax.scan(
                step,
                carry,
                length=self.cfg.algorithm.train_frequency
                // self.cfg.algorithm.num_envs,
            )

            key, state, loss, q_value = update(key, state)

            info["losses/loss"] = loss
            info["losses/q_value"] = q_value

            return (key, state), info

        (key, state), info = jax.lax.scan(
            learn,
            (key, state),
            length=(num_steps // self.cfg.algorithm.train_frequency),
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
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        state = state.replace(obs=obs, env_state=env_state)

        def step(carry: tuple[chex.PRNGKey, DQNState], _):
            key, state = carry

            q_values = self.q_network.apply(
                state.params,
                state.obs,
            )
            action = q_values.argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                reward=reward,  # type: ignore
                done=done,  # type: ignore
                info=info,  # type: ignore
            )

            state = state.replace(obs=next_obs, env_state=env_state)  # type: ignore

            return (key, state), transition

        (key, *_), transitions = jax.lax.scan(step, (key, state), length=num_steps)

        return key, transitions


def make_dqn(cfg, env, env_params, logger) -> DQN:
    """
    Factory function to construct a DQN agent from Args.

    Args:
        args: Experiment configuration.

    Returns:
        An initialized DQN instance ready for training.
    """

    q_network = Network(
        feature_extractor=instantiate(cfg.algorithm.feature_extractor),
        torso=instantiate(cfg.algorithm.torso),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )

    buffer = instantiate(cfg.algorithm.buffer)
    optimizer = optax.adam(learning_rate=cfg.algorithm.learning_rate)
    epsilon_schedule = optax.linear_schedule(
        cfg.algorithm.start_e,
        cfg.algorithm.end_e,
        int(cfg.algorithm.exploration_fraction * cfg.total_timesteps),
        cfg.algorithm.learning_starts,
    )
    return DQN(
        cfg=cfg,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
