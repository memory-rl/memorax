from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
from flax import core

from memory_rl.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
)
from memory_rl.networks import Network
from memory_rl.utils import Transition


@struct.dataclass(frozen=True)
class PQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    num_steps: int
    gamma: float
    td_lambda: float
    num_minibatches: int
    update_epochs: int
    td_lambda: float
    max_grad_norm: float
    learning_starts: int
    start_e: float
    end_e: float
    exploration_fraction: float
    feature_extractor: nn.Module
    torso: nn.Module

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PQNState:
    """
    Immutable container for training state of PQN algorithm.
    """

    step: int
    obs: Array
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState


@struct.dataclass(frozen=True)
class PQN:
    """
    Deep Q-Network (PQN) reinforcement learning algorithm.
    """

    cfg: PQNConfig
    env: Environment
    env_params: EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(self, key: Key, state: PQNState) -> tuple[Key, Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action, q_values

    def _random_action(self, key: Key, state: PQNState) -> tuple[Key, Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action, None

    def _epsilon_greedy_action(self, key: Key, state: PQNState) -> tuple[Key, Array]:

        key, random_action, _ = self._random_action(key, state)

        key, greedy_action, q_values = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, action, q_values

    def _step(self, carry, _, *, policy: Callable) -> tuple[Key, PQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action, q_values = policy(action_key, state)
        num_envs = state.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            done=done,  # type: ignore
            next_obs=next_obs,  # type: ignore
            info=info,  # type: ignore
            value=q_values,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    def _lambda_backscan(self, carry, transition):
        lambda_return, next_q_value = carry

        target_bootstrap = (
            transition.reward + self.cfg.gamma * (1.0 - transition.done) * next_q_value
        )

        delta = lambda_return - next_q_value
        lambda_return = target_bootstrap + self.cfg.gamma * self.cfg.td_lambda * delta

        lambda_return = (
            1.0 - transition.done
        ) * lambda_return + transition.done * transition.reward

        q_value = jnp.max(transition.value, axis=-1)
        return (lambda_return, q_value), lambda_return

    def _preprocess_transition(self, key, x):
        x = x.reshape(-1, *x.shape[2:])
        x = jax.random.permutation(key, x)
        x = x.reshape(self.cfg.num_minibatches, -1, *x.shape[1:])
        return x

    def _update_epoch(self, carry, _):
        key, state, transitions, lambda_targets = carry

        key, split_key = jax.random.split(key)
        minibatches = jax.tree_util.tree_map(
            lambda x: self._preprocess_transition(split_key, x), transitions
        )
        targets = self._preprocess_transition(split_key, lambda_targets)

        (key, state), (loss, q_value) = jax.lax.scan(
            self._update_minibatch, (key, state), xs=(minibatches, targets)
        )
        return (key, state, transitions, lambda_targets), (loss, q_value)

    def _update_minibatch(self, carry, xs) -> tuple[PQNState, Array, Array]:
        key, state = carry
        minibatch, target = xs

        def loss_fn(params):
            q_value = self.q_network.apply(
                params,
                minibatch.obs,
            )
            action = jnp.expand_dims(minibatch.action, axis=-1)
            q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
            loss = 0.5 * jnp.square(q_value - target).mean()
            return loss, q_value

        (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        state = state.replace(
            params=params,
            optimizer_state=optimizer_state,
        )

        return (key, state), (loss, q_value.mean())

    def _learn(
        self, carry: tuple[Key, PQNState], _
    ) -> tuple[tuple[Key, PQNState], dict]:

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.num_steps,
        )

        final_next_obs = transitions.next_obs[-1]
        final_done = transitions.done[-1]
        final_reward = transitions.reward[-1]

        final_q_values = self.q_network.apply(state.params, final_next_obs)
        final_q_value = jnp.max(final_q_values, axis=-1) * (1.0 - final_done)

        lambda_returns = final_reward + self.cfg.gamma * final_q_value
        _, targets = jax.lax.scan(
            self._lambda_backscan,
            (lambda_returns, final_q_value),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        (key, state, transitions, _), (loss, q_value) = jax.lax.scan(
            self._update_epoch,
            (key, state, transitions, lambda_targets),
            None,
            self.cfg.update_epochs,
        )

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[Key, PQNState, Array, EnvState]:
        """
        Initialize environment, network parameters, optimizer, and replay buffer.

        Args:
            key: JAX PRNG key for randomness.

        Returns:
            key: Updated PRNG key after splits.
            state: Initialized PQNState with params, target_params, optimizer_state, buffer_state.
            obs: Initial observations from vectorized envs.
            env_state: Initial environment state.
        """
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        return (
            key,
            PQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: PQNState, num_steps: int) -> tuple[Key, PQNState]:
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: Key,
        state: PQNState,
        num_steps: int,
    ) -> tuple[Key, PQNState, dict]:
        """
        Run training loop for specified number of steps.

        Args:
            key: JAX PRNG key.
            state: Current PQNState.
            obs: Current env observations.
            env_state: Current env state.
            num_steps: Total environment steps to train.

        Returns:
            key: Updated PRNG key.
            state: Updated PQNState after training.
            obs: Latest observations.
            env_state: Latest env state.
            info: Training statistics (loss, rewards, etc.).
        """
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // (self.cfg.num_steps * self.cfg.num_envs)),
        )

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, -1, 1), transitions)
        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: PQNState, num_steps: int) -> tuple[Key, dict]:
        """
        Evaluate current policy for a fixed number of steps without exploration.

        Args:
            key: JAX PRNG key.
            state: PQNState with trained parameters.
            num_steps: Number of evaluation steps.

        Returns:
            key: Updated PRNG key.
            info: Evaluation metrics (rewards, episode lengths, etc.).
        """
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        state = state.replace(obs=obs, env_state=env_state)

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
