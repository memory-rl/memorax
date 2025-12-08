from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
import optax
from flax import core

from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
)
from memorax.networks import Network
from memorax.utils import Transition


@struct.dataclass(frozen=True)
class DQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    gamma: float
    tau: float
    target_network_frequency: int
    batch_size: int
    start_e: float
    end_e: float
    exploration_fraction: float
    learning_starts: int
    train_frequency: int
    double: bool
    per_beta: Optional[float] = None
    per_epsilon: float = 1e-6


@struct.dataclass(frozen=True)
class DQNState:
    """
    Immutable container for training state of DQN algorithm.
    """

    step: int
    obs: Array
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


@struct.dataclass(frozen=True)
class DQN:
    """
    Deep Q-Network (DQN) reinforcement learning algorithm.
    """

    cfg: DQNConfig
    env: Environment
    env_params: EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule

    def _greedy_action(self, key: Key, state: DQNState) -> tuple[Key, Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action

    def _random_action(self, key: Key, state: DQNState) -> tuple[Key, Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action

    def _epsilon_greedy_action(self, key: Key, state: DQNState) -> tuple[Key, Array]:

        key, random_action = self._random_action(key, state)

        key, greedy_action = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, action

    def _step(
        self, carry, _, *, policy: Callable, write_to_buffer: bool = True
    ) -> tuple[Key, DQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action = policy(action_key, state)

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
            info=info,  # type: ignore
        )

        buffer_state = state.buffer_state

        if write_to_buffer:
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(self, key: Key, state: DQNState) -> tuple[DQNState, Array, Array]:

        key, sample_key = jax.random.split(key)
        sample = self.buffer.sample(state.buffer_state, sample_key)
        batch = sample.experience

        def make_dqn_target():
            next_target_q_values = self.q_network.apply(
                state.target_params,
                batch.second.obs,
            )
            return jnp.max(next_target_q_values, axis=-1)

        def make_double_dqn_target():
            next_q_values = self.q_network.apply(
                state.params,
                batch.second.obs,
            )
            next_action = jnp.argmax(next_q_values, axis=-1)
            next_target_q_value = jnp.take_along_axis(
                next_target_q_values, jnp.expand_dims(next_action, -1), axis=-1
            ).squeeze(-1)
            return next_target_q_value

        next_target_q_value = (
            make_dqn_target() if not self.cfg.double else make_double_dqn_target()
        )

        td_target = (
            batch.first.reward
            + (1 - batch.first.done) * self.cfg.gamma * next_target_q_value
        )

        if self.cfg.per_beta is not None:
            probs = jnp.maximum(sample.priorities, 1e-12)
            w = (1.0 / probs) ** self.cfg.per_beta
            w = w / jnp.max(w)
        else:
            w = 1.0

        def loss_fn(params):
            q_value = self.q_network.apply(
                params,
                batch.first.obs,
            )
            action = jnp.expand_dims(batch.first.action, axis=-1)
            q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
            td_error = q_value - td_target
            loss = (w * jnp.square(td_error)).mean()
            return loss, (q_value, td_error)

        (loss, (q_value, td_error)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        target_params = optax.periodic_update(
            optax.incremental_update(params, state.target_params, self.cfg.tau),
            state.target_params,
            state.step,
            self.cfg.target_network_frequency,
        )

        buffer_state = state.buffer_state
        if self.cfg.per_beta is not None:
            priorities = jnp.abs(td_error) + self.cfg.per_epsilon
            buffer_state = self.buffer.set_priorities(
                state.buffer_state, sample.indices, priorities
            )

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return key, state, loss, q_value.mean()

    def _learn(
        self, carry: tuple[Key, DQNState], _
    ) -> tuple[tuple[Key, DQNState], dict]:

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, state, loss, q_value = self._update(key, state)

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[Key, DQNState, Array, EnvState]:
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        params = self.q_network.init(q_key, obs)
        target_params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        transition = Transition(obs=obs, action=action, reward=reward, done=done, info=info)  # type: ignore
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

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
    def warmup(self, key: Key, state: DQNState, num_steps: int) -> tuple[Key, DQNState]:

        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: Key,
        state: DQNState,
        num_steps: int,
    ) -> tuple[Key, DQNState, dict]:
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: DQNState, num_steps: int) -> tuple[Key, Any]:
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        state = state.replace(obs=obs, env_state=env_state)

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
