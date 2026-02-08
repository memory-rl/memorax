from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import core
from flax import linen as nn
from flax import struct

from memorax.networks.sequence_models.utils import (add_feature_axis,
                                                    add_time_axis,
                                                    remove_feature_axis,
                                                    remove_time_axis)
from memorax.utils import Timestep, Transition
from memorax.utils.typing import Array, Key

to_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(add_time_axis)(x), timestep
)

from_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(remove_time_axis)(x), timestep
)


@struct.dataclass(frozen=True)
class MAPQNConfig:
    name: str
    num_envs: int
    num_eval_envs: int
    num_steps: int
    gamma: float
    td_lambda: float
    num_minibatches: int
    update_epochs: int
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class MAPQNState:
    step: int
    timestep: Timestep
    env_state: Any
    params: core.FrozenDict[str, Any]
    hidden_state: Array
    optimizer_state: optax.OptState


@struct.dataclass(frozen=True)
class MAPQN:
    cfg: MAPQNConfig
    env: Any
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    @property
    def _env_vectorized(self) -> bool:
        return getattr(self.env, "vectorized", False)

    def _env_reset(self, keys: Array, num_envs: int):
        if self._env_vectorized:
            obs, env_state = self.env.reset(keys[0])
            return obs, env_state
        else:
            return jax.vmap(self.env.reset, out_axes=(1, 0))(keys)

    def _env_step(self, keys: Array, env_state, actions: Array):
        if self._env_vectorized:
            return self.env.step(keys[0], env_state, actions)
        else:
            return jax.vmap(
                self.env.step, in_axes=(0, 0, 1), out_axes=(1, 0, 1, 1, 0)
            )(keys, env_state, actions)

    def _greedy_action(
        self, key: Key, state: MAPQNState
    ) -> tuple[Key, MAPQNState, Array, Array]:
        timestep = to_sequence(state.timestep)
        hidden_state, (q_values, _) = self.q_network.apply(
            state.params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.hidden_state,
        )
        q_values = jax.vmap(remove_time_axis)(q_values)
        action = jnp.argmax(q_values, axis=-1)
        state = state.replace(hidden_state=hidden_state)
        return key, state, action, q_values

    def _random_action(
        self, key: Key, state: MAPQNState
    ) -> tuple[Key, MAPQNState, Array, None]:
        key, action_key = jax.random.split(key)
        action_space = self.env.action_spaces[self.env.agents[0]]
        action_keys = jax.random.split(action_key, self.env.num_agents * self.cfg.num_envs)
        action_keys = action_keys.reshape(self.env.num_agents, self.cfg.num_envs)
        action = jax.vmap(jax.vmap(action_space.sample))(action_keys)
        return key, state, action, None

    def _epsilon_greedy_action(
        self, key: Key, state: MAPQNState
    ) -> tuple[Key, MAPQNState, Array, Array]:
        key, state, random_action, _ = self._random_action(key, state)
        key, state, greedy_action, q_values = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, state, action, q_values

    def _step(self, carry: tuple, _, *, policy: Callable):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, q_values = policy(action_key, state)

        _, num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = self._env_step(
            step_keys, state.env_state, action
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        prev_action = jnp.where(
            jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        prev_reward = jnp.where(state.timestep.done, 0, state.timestep.reward)

        transition = Transition(
            obs=state.timestep.obs,
            action=action,
            reward=reward,
            done=done,
            info=info,
            value=q_values,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=state.timestep.done,
        )

        state = state.replace(
            step=state.step + num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            env_state=env_state,
        )
        return (key, state), transition

    def _td_lambda(self, carry, transition):
        lambda_return, next_q_value = carry

        reward = transition.reward
        done = transition.done

        target_bootstrap = (
            reward + self.cfg.gamma * (1.0 - done) * next_q_value
        )

        delta = lambda_return - next_q_value
        lambda_return = target_bootstrap + self.cfg.gamma * self.cfg.td_lambda * delta
        lambda_return = (1.0 - done) * lambda_return + done * reward

        q_value = jnp.max(transition.value, axis=-1)
        return (lambda_return, q_value), lambda_return

    def _update_epoch(self, carry, _):
        key, state, initial_hidden_state, transitions, lambda_targets = carry

        key, permutation_key = jax.random.split(key)
        batch = (initial_hidden_state, transitions, lambda_targets)

        def shuffle(batch):
            shuffle_time_axis = initial_hidden_state is None
            num_agents = self.env.num_agents
            num_envs = self.cfg.num_envs
            num_steps = self.cfg.num_steps

            if shuffle_time_axis:
                batch = (
                    initial_hidden_state,
                    *jax.tree.map(
                        lambda x: x.reshape(num_agents, -1, 1, *x.shape[3:]),
                        (transitions, lambda_targets),
                    ),
                )
                num_samples_per_agent = num_envs * num_steps
            else:
                num_samples_per_agent = num_envs

            permutation = jax.random.permutation(permutation_key, num_samples_per_agent)

            minibatches = jax.tree.map(
                lambda x: (
                    jnp.moveaxis(
                        jnp.take(x, permutation, axis=1).reshape(
                            num_agents, self.cfg.num_minibatches, -1, *x.shape[2:]
                        ),
                        1, 0,
                    )
                    if x is not None
                    else None
                ),
                tuple(batch),
            )
            return minibatches

        minibatches = shuffle(batch)

        (key, state), metrics = jax.lax.scan(
            self._update_minibatch, (key, state), xs=minibatches
        )
        return (key, state, initial_hidden_state, transitions, lambda_targets), metrics

    def _update_minibatch(
        self, carry, minibatch
    ) -> tuple[tuple[MAPQNState, Array], tuple[Array, Array]]:
        key, state = carry

        hidden_state, transitions, target = minibatch

        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )
            hidden_state, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.prev_action,
                add_feature_axis(burn_in.prev_reward),
                burn_in.prev_done,
                hidden_state,
            )
            hidden_state = jax.lax.stop_gradient(hidden_state)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            target = target[:, :, self.cfg.burn_in_length :]

        def loss_fn(params):
            _, (q_values, aux) = self.q_network.apply(
                params,
                transitions.obs,
                transitions.prev_done,
                transitions.prev_action,
                add_feature_axis(transitions.prev_reward),
                transitions.prev_done,
                hidden_state,
                rngs={"memory": memory_key, "dropout": dropout_key},
            )
            action = add_feature_axis(transitions.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)

            td_error = q_value - target
            loss = 0.5 * jnp.square(td_error).mean()
            return loss, (
                q_value.mean(),
                q_value.min(),
                q_value.max(),
                q_value.std(),
                jnp.abs(td_error).mean(),
                td_error.std(),
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        state = state.replace(
            params=params,
            optimizer_state=optimizer_state,
        )

        return (key, state), (loss, *aux)

    def _learn(
        self, carry: tuple[Key, MAPQNState], _
    ) -> tuple[tuple[Key, MAPQNState], dict]:
        key, state = carry

        initial_hidden_state = state.hidden_state
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.num_steps,
        )

        key, memory_key, dropout_key = jax.random.split(key, 3)

        timestep = to_sequence(state.timestep)
        _, (q_values, _) = self.q_network.apply(
            state.params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.hidden_state,
            rngs={"memory": memory_key, "dropout": dropout_key},
        )
        q_value = jnp.max(q_values, axis=-1) * (1.0 - timestep.done)
        q_value = jax.vmap(remove_time_axis)(q_value)

        _, targets = jax.lax.scan(
            self._td_lambda,
            (q_value, q_value),
            transitions,
            reverse=True,
        )

        # (time, agents, envs, ...) -> (agents, envs, time, ...)
        transitions = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, min(2, x.ndim - 1)), transitions
        )
        targets = jnp.moveaxis(targets, 0, 2)

        (key, state, _, transitions, _), metrics = jax.lax.scan(
            self._update_epoch,
            (key, state, initial_hidden_state, transitions, targets),
            None,
            self.cfg.update_epochs,
        )

        loss, q_value, q_value_min, q_value_max, q_value_std, td_error, td_error_std = (
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=(0, 1, 2)), metrics)
        )
        epsilon = jnp.expand_dims(self.epsilon_schedule(state.step), axis=(0, 1, 2))
        info = {
            **transitions.info,
            "losses/loss": loss,
            "losses/q_value": q_value,
            "losses/q_value_min": q_value_min,
            "losses/q_value_max": q_value_max,
            "losses/q_value_std": q_value_std,
            "losses/td_error": td_error,
            "losses/td_error_std": td_error_std,
            "losses/epsilon": epsilon,
        }

        return (key, state), transitions.replace(obs=None, next_obs=None, info=info)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, q_key, memory_key = jax.random.split(key, 4)

        num_agents = self.env.num_agents

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = self._env_reset(env_keys, self.cfg.num_envs)

        action_space = self.env.action_spaces[self.env.agents[0]]

        action = jnp.zeros(
            (num_agents, self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_envs), dtype=jnp.bool)

        timestep = to_sequence(Timestep(
            obs=obs, action=action, reward=reward, done=done
        ))

        hidden_state = self.q_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )

        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            hidden_state,
        )
        optimizer_state = self.optimizer.init(params)

        return (
            key,
            MAPQNState(
                step=0,
                timestep=from_sequence(timestep),
                hidden_state=hidden_state,
                env_state=env_state,
                params=params,
                optimizer_state=optimizer_state,
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: MAPQNState, num_steps: int) -> tuple[Key, MAPQNState]:
        return key, state

    @partial(jax.jit, static_argnums=(0, 3))
    def train(self, key, state, num_steps):
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=num_steps // (self.cfg.num_envs * self.cfg.num_steps),
        )
        transitions = jax.tree.map(
            lambda x: jnp.moveaxis(x, 3, 1).reshape(-1, x.shape[1], x.shape[2], *x.shape[4:]),
            transitions,
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: MAPQNState, num_steps: int) -> tuple[Key, dict]:
        key, reset_key = jax.random.split(key)
        num_agents = self.env.num_agents

        reset_keys = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = self._env_reset(reset_keys, self.cfg.num_eval_envs)

        action_space = self.env.action_spaces[self.env.agents[0]]
        action = jnp.zeros(
            (num_agents, self.cfg.num_eval_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_eval_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_eval_envs), dtype=jnp.bool)

        hidden_state = self.q_network.initialize_carry(
            (num_agents, self.cfg.num_eval_envs, None)
        )

        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            env_state=env_state,
            hidden_state=hidden_state,
        )

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
