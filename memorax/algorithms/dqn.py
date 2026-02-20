from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.networks.sequence_models.utils import (
    add_feature_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import (
    Timestep,
    Transition,
    memory_metrics,
    periodic_incremental_update,
)
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
)


@struct.dataclass(frozen=True)
class DQNConfig:
    num_envs: int
    buffer_size: int
    tau: float
    target_update_frequency: int
    batch_size: int
    start_e: float
    end_e: float
    exploration_fraction: float
    train_frequency: int
    burn_in_length: int = 0


@struct.dataclass(frozen=True)
class DQNState:
    step: int
    timestep: Timestep
    carry: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


@struct.dataclass(frozen=True)
class DQN:
    cfg: DQNConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: DQNState
    ) -> tuple[Key, DQNState, Array, dict]:
        key, memory_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.carry,
            rngs={"memory": memory_key},
            mutable=["intermediates"],
        )
        action = jnp.argmax(q_values, axis=-1)
        action = remove_time_axis(action)
        state = state.replace(carry=carry)
        return key, state, action, intermediates

    def _random_action(
        self, key: Key, state: DQNState
    ) -> tuple[Key, DQNState, Array, dict]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action, {}

    def _epsilon_greedy_action(
        self, key: Key, state: DQNState
    ) -> tuple[Key, DQNState, Array, dict]:
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

    def _step(self, carry, _, *, policy: Callable, write_to_buffer: bool = True):
        key, state = carry

        initial_carry = state.carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, intermediates = policy(action_key, state)
        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(x, axis=(1, 2)),
            intermediates.get("intermediates", {}),
        )

        prev_action = jnp.where(
            state.timestep.done,
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        prev_reward = jnp.where(state.timestep.done, 0, state.timestep.reward)

        first = Timestep(
            obs=state.timestep.obs,
            action=prev_action,
            reward=prev_reward,
            done=state.timestep.done,
        )
        second = Timestep(
            obs=next_obs,
            action=action,
            reward=reward,
            done=done,
        )
        transition = Transition(
            first=first,
            second=second,
            metadata={**info, "intermediates": intermediates},
            carry=initial_carry,
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(self, key: Key, state: DQNState):
        batch = self.buffer.sample(state.buffer_state, key)

        key, memory_key, next_memory_key = jax.random.split(key, 3)

        experience = batch.experience
        experience = jax.tree.map(lambda x: jnp.expand_dims(x, 1), experience)

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
                observation=burn_in.first.obs,
                mask=burn_in.first.done,
                action=burn_in.first.action,
                reward=add_feature_axis(burn_in.first.reward),
                done=burn_in.first.done,
                initial_carry=initial_carry,
            )
            initial_carry = jax.lax.stop_gradient(initial_carry)
            initial_target_carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.target_params),
                observation=burn_in.second.obs,
                mask=burn_in.second.done,
                action=burn_in.second.action,
                reward=add_feature_axis(burn_in.second.reward),
                done=burn_in.second.done,
                initial_carry=initial_target_carry,
            )
            initial_target_carry = jax.lax.stop_gradient(initial_target_carry)
            experience = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], experience
            )

        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            observation=experience.second.obs,
            mask=experience.second.done,
            action=experience.second.action,
            reward=add_feature_axis(experience.second.reward),
            done=experience.second.done,
            initial_carry=initial_target_carry,
            rngs={"memory": next_memory_key},
        )
        next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        td_target = self.q_network.head.get_target(experience, next_target_q_value)

        def loss_fn(params):
            carry, (q_values, aux) = self.q_network.apply(
                params,
                observation=experience.first.obs,
                mask=experience.first.done,
                action=experience.first.action,
                reward=add_feature_axis(experience.first.reward),
                done=experience.first.done,
                initial_carry=initial_carry,
                rngs={"memory": memory_key},
            )
            action = add_feature_axis(experience.second.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target
            loss = self.q_network.head.loss(q_value, aux, td_target, transitions=experience).mean()
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
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        info = {"losses/loss": loss, "losses/q_value": q_value.mean()}

        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, -1], experience.carry)
            info.update(memory_metrics(carry, initial_carry))

        buffer_state = state.buffer_state

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return state, info

    def _update_step(self, carry, _):
        key, state = carry
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, update_key = jax.random.split(key)
        state, info = self._update(update_key, state)

        metadata = {
            **transitions.metadata,
            **jax.tree.map(lambda x: jnp.expand_dims(x, axis=(0, 1)), info),
        }

        return (key, state), transitions.replace(
            first=transitions.first.replace(obs=None),
            second=transitions.second.replace(obs=None),
            metadata=metadata,
        )

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        *_, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        carry = self.q_network.initialize_carry(obs.shape)

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=carry,
        )
        target_params = params
        optimizer_state = self.optimizer.init(params)

        _, intermediates = self.q_network.apply(
            params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=carry,
            rngs={"memory": memory_key},
            mutable=["intermediates"],
        )
        intermediates = jax.tree.map(
            lambda x: jnp.mean(x, axis=(1, 2)),
            intermediates.get("intermediates", {}),
        )

        dummy_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        transition = Transition(
            first=dummy_timestep,
            second=dummy_timestep,
            metadata={**info, "intermediates": intermediates},
            carry=carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return (
            key,
            DQNState(
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
    ):
        (key, state), transitions = jax.lax.scan(
            self._update_step,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )

        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: DQNState, num_steps: int):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry = self.q_network.initialize_carry(obs.shape)

        state = state.replace(timestep=timestep, carry=carry, env_state=env_state)
        (key, _), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(
            first=transitions.first.replace(obs=None),
            second=transitions.second.replace(obs=None),
        )
