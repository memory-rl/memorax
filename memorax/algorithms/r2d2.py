"""R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning."""

from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.buffers import compute_importance_weights
from memorax.networks.sequence_models.utils import (add_feature_axis,
                                                    remove_feature_axis,
                                                    remove_time_axis)
from memorax.utils import Timestep, Transition, memory_metrics, periodic_incremental_update
from memorax.utils.typing import (Array, Buffer, BufferState, Environment,
                                  EnvParams, EnvState, Key)


@struct.dataclass(frozen=True)
class R2D2Config:
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    tau: float
    target_network_frequency: int
    batch_size: int
    start_e: float
    end_e: float
    exploration_fraction: float
    train_frequency: int
    burn_in_length: int = 10
    sequence_length: int = 80
    n_step: int = 5
    priority_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6
    double: bool = True


@struct.dataclass(frozen=True)
class R2D2State:
    step: int
    timestep: Timestep
    carry: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


def compute_n_step_returns(
    rewards: Array,
    dones: Array,
    next_q_values: Array,
    n_step: int,
    gamma: float,
) -> Array:
    batch_size, seq_len = rewards.shape
    num_targets = seq_len - n_step + 1

    def compute_target(start_idx):
        n_step_return = jnp.zeros(batch_size)
        discount = 1.0
        done = jnp.ones(batch_size)

        for i in range(n_step):
            idx = start_idx + i
            n_step_return = n_step_return + discount * rewards[:, idx] * done
            discount = discount * gamma
            done = done * (1.0 - dones[:, idx])

        bootstrap_idx = start_idx + n_step - 1
        n_step_return = (
            n_step_return + discount * next_q_values[:, bootstrap_idx] * done
        )

        return n_step_return

    targets = jax.vmap(compute_target)(jnp.arange(num_targets))
    targets = targets.T

    return targets


@struct.dataclass(frozen=True)
class R2D2:
    cfg: R2D2Config
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule
    beta_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[Key, R2D2State, Array, dict]:
        key, memory_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.carry,
            rngs={"memory": memory_key},
            mutable=['intermediates'],
        )
        action = jnp.argmax(q_values, axis=-1)
        action = remove_time_axis(action)
        state = state.replace(carry=carry)
        return key, state, action, intermediates

    def _random_action(
        self, key: Key, state: R2D2State
    ) -> tuple[Key, R2D2State, Array, dict]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action, {}

    def _epsilon_greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[Key, R2D2State, Array, dict]:
        key, state, random_action, _ = self._random_action(key, state)
        key, state, greedy_action, intermediates = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, state, action, intermediates

    def _step(
        self, carry, _, *, policy: Callable, write_to_buffer: bool = True
    ) -> tuple[Key, R2D2State]:
        key, state = carry

        initial_carry = state.carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, intermediates = policy(action_key, state)
        num_envs = state.timestep.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(x, axis=(1, 2)),
            intermediates.get('intermediates', {}),
        )

        prev_action = jnp.where(
            state.timestep.done,
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        prev_reward = jnp.where(state.timestep.done, 0, state.timestep.reward)

        transition = Transition(
            obs=state.timestep.obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info={**info, "intermediates": intermediates},
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=state.timestep.done,
            carry=initial_carry,
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=jnp.asarray(reward, dtype=jnp.float32), done=done),
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, Array, Array]:
        key, sample_key = jax.random.split(key)
        batch = self.buffer.sample(state.buffer_state, sample_key)

        key, memory_key, next_memory_key = jax.random.split(key, 3)

        experience = batch.experience

        initial_carry = None
        initial_target_carry = None
        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)
            initial_target_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], experience
            )
            initial_carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.prev_action,
                add_feature_axis(burn_in.prev_reward),
                burn_in.prev_done,
                initial_carry,
            )
            initial_carry = jax.lax.stop_gradient(initial_carry)
            initial_target_carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.target_params),
                burn_in.next_obs,
                burn_in.done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.done,
                initial_target_carry,
            )
            initial_target_carry = jax.lax.stop_gradient(initial_target_carry)
            experience = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], experience
            )

        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            experience.next_obs,
            experience.done,
            experience.action,
            add_feature_axis(experience.reward),
            experience.done,
            initial_target_carry,
            rngs={"memory": next_memory_key},
        )

        if self.cfg.double:
            _, (online_next_q_values, _) = self.q_network.apply(
                state.params,
                experience.next_obs,
                experience.done,
                experience.action,
                add_feature_axis(experience.reward),
                experience.done,
                initial_carry,
                rngs={"memory": memory_key},
            )
            next_actions = jnp.argmax(online_next_q_values, axis=-1)
            next_target_q_value = jnp.take_along_axis(
                next_target_q_values, add_feature_axis(next_actions), axis=-1
            )
            next_target_q_value = remove_feature_axis(next_target_q_value)
        else:
            next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        learning_seq_len = experience.reward.shape[1]
        if self.cfg.n_step > 1 and learning_seq_len >= self.cfg.n_step:
            n_step_targets = compute_n_step_returns(
                experience.reward,
                experience.done,
                next_target_q_value,
                self.cfg.n_step,
                self.q_network.head.gamma,
            )
            num_targets = n_step_targets.shape[1]
            experience = jax.tree.map(lambda x: x[:, :num_targets], experience)
            td_target = n_step_targets
        else:
            td_target = self.q_network.head.get_target(experience, next_target_q_value)

        beta = self.beta_schedule(state.step)
        buffer_size = jnp.where(
            state.buffer_state.is_full,
            self.cfg.buffer_size,
            state.buffer_state.current_index * self.cfg.num_envs,
        )
        buffer_size = jnp.maximum(buffer_size, 1)
        importance_weights = compute_importance_weights(
            batch.probabilities, buffer_size, beta
        )
        importance_weights = importance_weights[:, None]

        def loss_fn(params):
            carry, (q_values, aux) = self.q_network.apply(
                params,
                experience.obs,
                experience.prev_done,
                experience.prev_action,
                add_feature_axis(experience.prev_reward),
                experience.prev_done,
                initial_carry,
                rngs={"memory": memory_key},
            )
            action = add_feature_axis(experience.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target

            loss = (importance_weights * self.q_network.head.loss(q_value, aux, td_target, experience)).mean()
            return loss, (q_value, td_error, carry)

        (loss, (q_value, td_error, carry)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        target_params = periodic_incremental_update(
            params,
            state.target_params,
            state.step,
            self.cfg.target_network_frequency,
            self.cfg.tau,
        )

        mean_td_error = jnp.abs(td_error).mean(axis=1)
        new_priorities = mean_td_error + 1e-6
        buffer_state = self.buffer.set_priorities(
            state.buffer_state, batch.indices, new_priorities
        )

        info = {
            "losses/loss": loss,
            "losses/q_value": q_value.mean(),
            "losses/td_error": mean_td_error.mean(),
        }

        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, -1], experience.carry)
            info.update(memory_metrics(carry, initial_carry))

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return state, info

    def _learn(self, carry, _):
        key, state = carry
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, update_key = jax.random.split(key)
        state, info = self._update(update_key, state)

        info = {
            **transitions.info,
            **jax.tree.map(lambda x: jnp.expand_dims(x, axis=(0, 1)), info),
        }

        return (key, state), transitions.replace(obs=None, next_obs=None, info=info)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_envs,), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_envs, dtype=jnp.bool)
        *_, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        carry = self.q_network.initialize_carry(obs.shape)

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            carry,
        )
        target_params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            carry,
        )
        optimizer_state = self.optimizer.init(params)

        _, intermediates = self.q_network.apply(
            params, timestep.obs, timestep.done, timestep.action,
            add_feature_axis(timestep.reward), timestep.done, carry,
            rngs={"memory": memory_key}, mutable=['intermediates'],
        )
        intermediates = jax.tree.map(
            lambda x: jnp.mean(x, axis=(1, 2)),
            intermediates.get('intermediates', {}),
        )

        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=obs,
            done=done,
            info={**info, "intermediates": intermediates},
            prev_action=action,
            prev_reward=reward,
            prev_done=done,
            carry=carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return (
            key,
            R2D2State(
                step=0,
                timestep=timestep.from_sequence(),
                carry=carry,
                env_state=env_state,
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
                buffer_state=buffer_state,
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: Key, state: R2D2State, num_steps: int
    ) -> tuple[Key, R2D2State]:
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
        state: R2D2State,
        num_steps: int,
    ):
        (
            (
                key,
                state,
            ),
            transitions,
        ) = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )

        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: R2D2State, num_steps: int) -> tuple[Key, dict]:
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_eval_envs,), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_eval_envs, dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry = self.q_network.initialize_carry(obs.shape)

        state = state.replace(
            timestep=timestep, carry=carry, env_state=env_state
        )

        (key, _), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
