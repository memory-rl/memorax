from functools import partial
from typing import Any, Callable

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.networks import SequenceModelWrapper
from memorax.networks.sequence_models.utils import (
    add_feature_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import Timestep, Transition
from memorax.utils.typing import Array, Environment, EnvParams, EnvState, Key


@struct.dataclass(frozen=True)
class PQNConfig:
    name: str
    num_envs: int
    num_eval_envs: int
    num_steps: int
    gamma: float
    td_lambda: float
    num_minibatches: int
    update_epochs: int

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PQNState:
    step: int
    timestep: Timestep
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    hidden_state: Array
    optimizer_state: optax.OptState


@struct.dataclass(frozen=True)
class PQN:
    cfg: PQNConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: PQNState
    ) -> tuple[Key, PQNState, Array, Array]:
        timestep = state.timestep.to_sequence()
        hidden_state, q_values = self.q_network.apply(
            state.params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.hidden_state,
        )
        q_values = remove_time_axis(q_values)
        action = jnp.argmax(q_values, axis=-1)
        state = state.replace(hidden_state=hidden_state)
        return key, state, action, q_values

    def _random_action(
        self, key: Key, state: PQNState
    ) -> tuple[Key, PQNState, Array, None]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action, None

    def _epsilon_greedy_action(
        self, key: Key, state: PQNState
    ) -> tuple[Key, PQNState, Array, Array]:
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

    def _step(
        self, carry, _, *, policy: Callable
    ) -> tuple[tuple[Key, PQNState], Transition]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, q_values = policy(action_key, state)
        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        prev_action = jnp.where(
            state.timestep.done,
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        prev_reward = jnp.where(
            state.timestep.done,
            jnp.zeros_like(state.timestep.reward),
            state.timestep.reward,
        )
        transition = Transition(
            obs=state.timestep.obs,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
            value=q_values,  # type: ignore
            next_obs=next_obs,  # type: ignore
            prev_action=prev_action,  # type: ignore
            prev_reward=prev_reward,  # type: ignore
            prev_done=state.timestep.done,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),  # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    def _td_lambda(self, carry, transition):
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

    def _update_epoch(self, carry, _):
        key, state, initial_hidden_state, transitions, lambda_targets = carry

        key, permutation_key = jax.random.split(key)
        batch = (initial_hidden_state, transitions, lambda_targets)

        def shuffle(batch):
            shuffle_time_axis = isinstance(self.q_network, SequenceModelWrapper)
            num_permutations = self.cfg.num_envs
            if shuffle_time_axis:
                batch = (
                    initial_hidden_state,
                    *jax.tree.map(
                        lambda x: x.reshape(-1, 1, *x.shape[2:]),
                        (transitions, lambda_targets),
                    ),
                )
                num_permutations *= self.cfg.num_steps

            permutation = jax.random.permutation(permutation_key, num_permutations)

            minibatches = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(
                    self.cfg.num_minibatches, -1, *x.shape[1:]
                ),
                tuple(batch),
            )
            return minibatches

        minibatches = shuffle(batch)

        (key, state), (loss, q_value) = jax.lax.scan(
            self._update_minibatch, (key, state), xs=minibatches
        )
        return (key, state, initial_hidden_state, transitions, lambda_targets), (
            loss,
            q_value,
        )

    def _update_minibatch(
        self, carry, minibatch
    ) -> tuple[tuple[PQNState, Array], tuple[Array, Array]]:
        key, state = carry

        hidden_state, transitions, target = minibatch

        key, memory_key, dropout_key = jax.random.split(key, 3)

        def loss_fn(params):
            _, q_values = self.q_network.apply(
                params,
                observation=transitions.obs,
                mask=transitions.prev_done,
                action=transitions.prev_action,
                reward=add_feature_axis(transitions.prev_reward),
                done=transitions.prev_done,
                initial_carry=hidden_state,
                rngs={"memory": memory_key, "dropout": dropout_key},
            )
            action = add_feature_axis(transitions.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)

            loss = 0.5 * jnp.square(q_value - target).mean()
            return loss, q_value.mean()

        (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        state = state.replace(
            params=params,
            optimizer_state=optimizer_state,
        )

        return (key, state), (loss, q_value)

    def _learn(
        self, carry: tuple[Key, PQNState], _
    ) -> tuple[tuple[Key, PQNState], dict]:
        key, state = carry

        initial_hidden_state = state.hidden_state
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.num_steps,
        )

        key, memory_key, dropout_key = jax.random.split(key, 3)

        timestep = state.timestep.to_sequence()
        _, q_values = self.q_network.apply(
            state.params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.hidden_state,
            rngs={"memory": memory_key, "dropout": dropout_key},
        )
        q_value = jnp.max(q_values, axis=-1) * (1.0 - timestep.done)
        q_value = remove_time_axis(q_value)

        _, targets = jax.lax.scan(
            self._td_lambda,
            (q_value, q_value),
            transitions,
            reverse=True,
        )
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        targets = jnp.swapaxes(targets, 0, 1)

        (key, state, _, transitions, _), (loss, q_value) = jax.lax.scan(
            self._update_epoch,
            (key, state, initial_hidden_state, transitions, targets),
            None,
            self.cfg.update_epochs,
        )

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[Key, PQNState, Array, gymnax.EnvState]:
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.zeros(self.cfg.num_envs, dtype=jnp.bool)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        hidden_state = self.q_network.initialize_carry((self.cfg.num_envs, None))
        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=hidden_state,
        )
        optimizer_state = self.optimizer.init(params)

        return (
            key,
            PQNState(
                step=0,  # type: ignore
                timestep=timestep.from_sequence(),  # type: ignore
                hidden_state=hidden_state,  # type: ignore
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
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // (self.cfg.num_steps * self.cfg.num_envs)),
        )
        transitions = jax.tree.map(
            lambda x: x.swapaxes(1, 2).reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: PQNState, num_steps: int) -> tuple[Key, dict]:
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_eval_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.float32)
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=jnp.bool)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        hidden_state = self.q_network.initialize_carry((self.cfg.num_eval_envs, None))

        state = state.replace(
            timestep=timestep, hidden_state=hidden_state, env_state=env_state
        )

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
