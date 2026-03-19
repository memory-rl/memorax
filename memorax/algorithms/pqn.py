from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class PQNConfig:
    num_envs: int
    num_steps: int
    td_lambda: float
    num_minibatches: int
    update_epochs: int
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PQNState:
    step: int
    timestep: Timestep
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    carry: Array
    optimizer_state: optax.OptState


@dataclass
class PQN:
    cfg: PQNConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: PQNState
    ) -> tuple[PQNState, Array, Array, dict]:
        timestep = state.timestep.to_sequence()
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.carry,
            mutable=["intermediates"],
        )
        q_values = remove_time_axis(q_values)
        action = jnp.argmax(q_values, axis=-1)
        state = state.replace(carry=carry)
        return state, action, q_values, intermediates

    def _random_action(
        self, key: Key, state: PQNState
    ) -> tuple[PQNState, Array, None, dict]:
        action_key = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return state, action, None, {}

    def _epsilon_greedy_action(
        self, key: Key, state: PQNState
    ) -> tuple[PQNState, Array, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)

        state, random_action, _, _ = self._random_action(random_key, state)

        state, greedy_action, q_values, intermediates = self._greedy_action(
            greedy_key, state
        )

        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return state, action, q_values, intermediates

    def _step(
        self, state: PQNState, key: Key, *, policy: Callable
    ) -> tuple[PQNState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, q_values, intermediates = policy(action_key, state)
        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        first = Timestep(
            obs=state.timestep.obs,
            action=jnp.where(
                state.timestep.done,
                jnp.zeros_like(state.timestep.action),
                state.timestep.action,
            ),
            reward=jnp.where(state.timestep.done, 0, state.timestep.reward),
            done=state.timestep.done,
        )
        second = Timestep(
            obs=None,
            action=action,
            reward=reward,
            done=done,
        )
        lox.log({"info": info, "intermediates": intermediates})

        transition = Transition(
            first=first,
            second=second,
            aux={"q_values": q_values},
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
        )
        return state, transition

    def _td_lambda(self, carry: tuple, transition: Transition):
        lambda_return, next_q_value = carry
        target_bootstrap = self.q_network.head.get_target(transition, next_q_value)

        delta = lambda_return - next_q_value
        lambda_return = (
            target_bootstrap + self.q_network.head.gamma * self.cfg.td_lambda * delta
        )

        lambda_return = (
            1.0 - transition.second.done
        ) * lambda_return + transition.second.done * transition.second.reward

        q_value = jnp.max(transition.aux["q_values"], axis=-1)
        return (lambda_return, q_value), lambda_return

    def _update_epoch(self, carry: tuple, key: Key):
        state, initial_carry, transitions, lambda_targets = carry

        permutation_key, minibatch_key = jax.random.split(key)
        batch = (initial_carry, transitions, lambda_targets)

        def shuffle(batch: PyTree):
            shuffle_time_axis = initial_carry is None
            num_permutations = self.cfg.num_envs
            if shuffle_time_axis:
                batch = (
                    initial_carry,
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
        minibatch_keys = jax.random.split(minibatch_key, self.cfg.num_minibatches)

        state, metrics = jax.lax.scan(
            self._update_minibatch, state, (minibatches, minibatch_keys)
        )
        return (state, initial_carry, transitions, lambda_targets), metrics

    def _update_minibatch(self, state: PQNState, xs: tuple):
        minibatch, key = xs

        carry, transitions, target = minibatch

        torso_key, dropout_key = jax.random.split(key)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
            )
            carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.params),
                observation=burn_in.first.obs,
                done=burn_in.first.done,
                action=burn_in.first.action,
                reward=add_feature_axis(burn_in.first.reward),
                initial_carry=carry,
            )
            carry = jax.lax.stop_gradient(carry)
            transitions = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            target = target[:, self.cfg.burn_in_length :]

        def loss_fn(params: PyTree):
            _, (q_values, aux) = self.q_network.apply(
                params,
                observation=transitions.first.obs,
                done=transitions.first.done,
                action=transitions.first.action,
                reward=add_feature_axis(transitions.first.reward),
                initial_carry=carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            action = add_feature_axis(transitions.second.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)

            td_error = q_value - target
            loss = self.q_network.head.loss(
                q_value, aux, target, transitions=transitions
            ).mean()
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

        return state, (loss, *aux)

    def _update_step(self, state: PQNState, key: Key) -> tuple[PQNState, None]:
        step_key, torso_key, epoch_key = jax.random.split(key, 3)

        initial_carry = state.carry

        step_keys = jax.random.split(step_key, self.cfg.num_steps)
        state, transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            state,
            step_keys,
        )

        timestep = state.timestep.to_sequence()
        _, (q_values, _) = self.q_network.apply(
            state.params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.carry,
            rngs={"torso": torso_key},
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

        epoch_keys = jax.random.split(epoch_key, self.cfg.update_epochs)
        (state, _, transitions, _), metrics = jax.lax.scan(
            self._update_epoch,
            (state, initial_carry, transitions, targets),
            epoch_keys,
        )

        loss, q_value, q_value_min, q_value_max, q_value_std, td_error, td_error_std = (
            metrics
        )
        lox.log(
            {
                "losses/loss": loss,
                "losses/q_value": q_value,
                "losses/q_value_min": q_value_min,
                "losses/q_value_max": q_value_max,
                "losses/q_value_std": q_value_std,
                "losses/td_error": td_error,
                "losses/td_error_std": td_error_std,
                "losses/epsilon": self.epsilon_schedule(state.step),
            }
        )

        return state, None

    def init(self, key: Key) -> PQNState:
        env_key, q_key, torso_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_envs, dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))
        params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=carry,
        )
        optimizer_state = self.optimizer.init(params)

        return PQNState(
            step=0,
            timestep=timestep.from_sequence(),
            carry=carry,
            env_state=env_state,
            params=params,
            optimizer_state=optimizer_state,
        )

    def warmup(self, key: Key, state: PQNState, num_steps: int) -> PQNState:
        return state

    def train(
        self,
        key: Key,
        state: PQNState,
        num_steps: int,
    ) -> PQNState:
        num_outer_steps = num_steps // (self.cfg.num_steps * self.cfg.num_envs)
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(self, key: Key, state: PQNState, num_steps: int) -> PQNState:
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_envs, dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))

        state = state.replace(timestep=timestep, carry=carry, env_state=env_state)

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            step_keys,
        )

        return state
