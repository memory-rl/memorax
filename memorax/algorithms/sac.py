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
class SACConfig:
    num_envs: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    buffer_size: int
    tau: float
    train_frequency: int
    target_update_frequency: int
    batch_size: int
    initial_alpha: float
    target_entropy_scale: float
    max_grad_norm: float
    gradient_steps: int = 1
    burn_in_length: int = 0


@struct.dataclass(frozen=True)
class SACState:
    step: int
    timestep: Timestep
    env_state: EnvState
    buffer_state: BufferState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    critic_target_params: core.FrozenDict[str, Any]
    alpha_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    actor_carry: Array
    critic_carry: Array


@struct.dataclass(frozen=True)
class SAC:
    cfg: SACConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    alpha_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    buffer: Buffer

    def _deterministic_action(self, key, state: SACState):
        key, sample_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        (next_carry, (dist, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            temperature=0.0,
            mutable=["intermediates"],
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        state = state.replace(actor_carry=next_carry)
        return key, state, action, intermediates

    def _stochastic_action(self, key, state: SACState):
        key, sample_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        (next_carry, (dist, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            mutable=["intermediates"],
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        state = state.replace(actor_carry=next_carry)
        return key, state, action, intermediates

    def _random_action(self, key, state: SACState):
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_keys)
        return key, state, action, {}

    def _step(
        self,
        carry,
        _,
        *,
        policy: Callable,
        write_to_buffer: bool = True,
    ):
        key, state = carry
        initial_carry = state.actor_carry

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
            buffer_state = self.buffer.add(buffer_state, transition)

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

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)
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

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        actor_carry = self.actor_network.initialize_carry(obs.shape)
        actor_params = self.actor_network.init(
            actor_key,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=actor_carry,
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)

        critic_carry = self.critic_network.initialize_carry(obs.shape)
        critic_params = self.critic_network.init(
            critic_key,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=critic_carry,
        )
        critic_target_params = critic_params
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        *_, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        _, intermediates = self.actor_network.apply(
            actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=actor_carry,
            mutable=["intermediates"],
        )
        intermediates = jax.tree.map(
            lambda x: jnp.mean(x, axis=(1, 2)),
            intermediates.get("intermediates", {}),
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
            carry=actor_carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return (
            key,
            SACState(
                step=0,
                timestep=timestep.from_sequence(),
                actor_carry=actor_carry,
                critic_carry=critic_carry,
                env_state=env_state,
                buffer_state=buffer_state,
                actor_params=actor_params,
                critic_params=critic_params,
                critic_target_params=critic_target_params,
                alpha_params=alpha_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
                alpha_optimizer_state=alpha_optimizer_state,
            ),
        )

    def _update_alpha(
        self,
        key,
        state: SACState,
        experience,
        initial_actor_carry=None,
    ):
        action_dim, *_ = self.env.action_space(self.env_params).shape
        target_entropy = -self.cfg.target_entropy_scale * action_dim

        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            observation=experience.obs,
            mask=experience.prev_done,
            action=experience.prev_action,
            reward=add_feature_axis(experience.prev_reward),
            done=experience.prev_done,
            initial_carry=initial_actor_carry,
        )

        key, sample_key = jax.random.split(key)
        actions = dist.sample(seed=sample_key)
        log_probs = dist.log_prob(actions)

        def alpha_loss_fn(alpha_params):
            log_alpha = self.alpha_network.apply(alpha_params)
            alpha = jnp.exp(log_alpha)
            alpha_loss = (alpha * (-log_probs - target_entropy)).mean()
            return alpha_loss, {"losses/alpha": alpha, "losses/alpha_loss": alpha_loss}

        (_, info), grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
            state.alpha_params
        )
        updates, optimizer_state = self.alpha_optimizer.update(
            grads, state.alpha_optimizer_state, state.alpha_params
        )
        alpha_params = optax.apply_updates(state.alpha_params, updates)

        state = state.replace(
            alpha_params=alpha_params, alpha_optimizer_state=optimizer_state
        )

        return state, info

    def _update_actor(
        self,
        key,
        state: SACState,
        experience,
        initial_actor_carry=None,
        initial_critic_carry=None,
    ):
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        def actor_loss_fn(actor_params):
            carry, (dist, _) = self.actor_network.apply(
                actor_params,
                observation=experience.obs,
                mask=experience.prev_done,
                action=experience.prev_action,
                reward=add_feature_axis(experience.prev_reward),
                done=experience.prev_done,
                initial_carry=initial_actor_carry,
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, (qs, _) = self.critic_network.apply(
                state.critic_params,
                observation=experience.obs,
                mask=experience.prev_done,
                action=actions,
                reward=add_feature_axis(experience.prev_reward),
                done=experience.prev_done,
                initial_carry=initial_critic_carry,
            )
            q = jnp.minimum(*qs)
            actor_loss = (log_probs * alpha - remove_feature_axis(q)).mean()
            return actor_loss, (
                carry,
                {"losses/actor_loss": actor_loss, "losses/entropy": -log_probs.mean()},
            )

        (_, (carry, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor_params
        )
        updates, actor_optimizer_state = self.actor_optimizer.update(
            grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, updates)

        state = state.replace(
            actor_params=actor_params, actor_optimizer_state=actor_optimizer_state
        )

        return state, carry, info

    def _update_critic(
        self,
        key,
        state: SACState,
        experience,
        initial_actor_carry=None,
        initial_critic_carry=None,
        initial_target_critic_carry=None,
    ):
        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            observation=experience.next_obs,
            mask=experience.done,
            action=experience.action,
            reward=add_feature_axis(experience.reward),
            done=experience.done,
            initial_carry=initial_actor_carry,
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, (next_qs, _) = self.critic_network.apply(
            state.critic_target_params,
            observation=experience.next_obs,
            mask=experience.done,
            action=next_actions,
            reward=add_feature_axis(experience.reward),
            done=experience.done,
            initial_carry=initial_target_critic_carry,
        )
        next_q = jnp.minimum(*next_qs)

        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)
        next_value = remove_feature_axis(next_q) - alpha * next_log_probs
        target_q = self.critic_network.head.get_target(experience, next_value)

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            _, (qs, _) = self.critic_network.apply(
                critic_params,
                observation=experience.obs,
                mask=experience.prev_done,
                action=experience.action,
                reward=add_feature_axis(experience.prev_reward),
                done=experience.prev_done,
                initial_carry=initial_critic_carry,
            )
            q1, q2 = qs
            critic_loss = (
                self.critic_network.head.loss(q1, {}, target_q, experience).mean()
                + self.critic_network.head.loss(q2, {}, target_q, experience).mean()
            )

            return critic_loss, {
                "losses/critic_loss": critic_loss,
                "losses/q1": q1.mean(),
                "losses/q2": q2.mean(),
            }

        (_, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic_params
        )
        updates, critic_optimizer_state = self.critic_optimizer.update(
            grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, updates)

        critic_target_params = periodic_incremental_update(
            critic_params,
            state.critic_target_params,
            state.step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        state = state.replace(
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return state, info

    def _update(self, key, state: SACState):
        key, batch_key, critic_key, actor_key, alpha_key = jax.random.split(key, 5)
        experience = self.buffer.sample(state.buffer_state, batch_key).experience
        experience = jax.tree.map(lambda x: jnp.expand_dims(x, 1), experience)

        initial_actor_carry = None
        initial_critic_carry = None
        initial_target_critic_carry = None

        if experience.carry is not None:
            initial_actor_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], experience
            )
            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                observation=burn_in.obs,
                mask=burn_in.prev_done,
                action=burn_in.prev_action,
                reward=add_feature_axis(burn_in.prev_reward),
                done=burn_in.prev_done,
                initial_carry=initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)

            initial_critic_carry, _ = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                observation=burn_in.obs,
                mask=burn_in.prev_done,
                action=burn_in.prev_action,
                reward=add_feature_axis(burn_in.prev_reward),
                done=burn_in.prev_done,
                initial_carry=initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)

            initial_target_critic_carry, _ = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_target_params),
                observation=burn_in.next_obs,
                mask=burn_in.done,
                action=burn_in.action,
                reward=add_feature_axis(burn_in.reward),
                done=burn_in.done,
                initial_carry=initial_target_critic_carry,
            )
            initial_target_critic_carry = jax.lax.stop_gradient(
                initial_target_critic_carry
            )

            experience = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], experience
            )

        state, critic_info = self._update_critic(
            critic_key,
            state,
            experience,
            initial_actor_carry,
            initial_critic_carry,
            initial_target_critic_carry,
        )
        state, actor_carry, actor_info = self._update_actor(
            actor_key,
            state,
            experience,
            initial_actor_carry,
            initial_critic_carry,
        )
        state, alpha_info = self._update_alpha(
            alpha_key, state, experience, initial_actor_carry
        )

        info = {**critic_info, **actor_info, **alpha_info}

        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, -1], experience.carry)
            info.update(memory_metrics(actor_carry, initial_carry))

        return state, info

    def _update_step(self, carry, _):
        key, state = carry
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            (key, state),
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        def _gradient_step(carry, _):
            key, state = carry
            key, update_key = jax.random.split(key)
            state, update_info = self._update(update_key, state)
            return (key, state), update_info

        (key, state), update_info = jax.lax.scan(
            _gradient_step, (key, state), length=self.cfg.gradient_steps
        )
        update_info = jax.tree.map(lambda x: x.mean(axis=0), update_info)

        info = {
            **transitions.info,
            **jax.tree.map(lambda x: jnp.expand_dims(x, axis=(0, 1)), update_info),
        }

        return (key, state), transitions.replace(obs=None, next_obs=None, info=info)

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: SACState, num_steps: int) -> tuple[Key, SACState]:
        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: SACState, num_steps: int):
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
    def evaluate(self, key: Key, state: SACState, num_steps: int):
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
        carry = self.actor_network.initialize_carry(obs.shape)

        state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_carry=carry,
        )

        (key, *_), transitions = jax.lax.scan(
            partial(
                self._step,
                policy=self._deterministic_action,
                write_to_buffer=False,
            ),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
