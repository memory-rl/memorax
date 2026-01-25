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
from memorax.utils import Timestep, Transition, periodic_incremental_update
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
    """Configuration for SAC algorithm."""

    name: str
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    gamma: float
    tau: float
    train_frequency: int
    target_update_frequency: int
    batch_size: int
    initial_alpha: float
    target_entropy_scale: float
    learning_starts: int
    max_grad_norm: float
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
    actor_hidden_state: Array
    critic_hidden_state: Array


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
        next_hidden_state, (dist, _) = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_hidden_state,
            temperature=0.0,
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        return key, (action, next_hidden_state)

    def _stochastic_action(self, key, state: SACState):
        key, sample_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        next_hidden_state, (dist, _) = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_hidden_state,
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        return key, (action, next_hidden_state)

    def _random_action(self, key, state: SACState):
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_keys)
        return key, (action, state.actor_hidden_state)

    def _step(
        self,
        carry,
        _,
        *,
        policy: Callable[[Key, "SACState"], tuple[Key, tuple[Array, Array]]],
        write_to_buffer: bool = True,
    ):
        key, state = carry
        key, policy_key, env_key = jax.random.split(key, 3)

        key, (action, next_actor_hidden_state) = policy(policy_key, state)

        num_envs = state.timestep.obs.shape[0]
        env_keys = jax.random.split(env_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(env_keys, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.timestep.obs,  # type: ignore
            prev_done=state.timestep.done,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            next_obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            env_state=env_state,
            buffer_state=buffer_state,
            actor_hidden_state=next_actor_hidden_state,
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
        action = jnp.zeros((self.cfg.num_envs,) + action_space.shape, dtype=jnp.float32)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        actor_carry = self.actor_network.initialize_carry(obs.shape)
        actor_params = self.actor_network.init(
            actor_key,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            actor_carry,
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        actor_hidden_state = self.actor_network.initialize_carry(obs.shape)

        critic_carry = self.critic_network.initialize_carry(obs.shape)
        critic_params = self.critic_network.init(
            critic_key,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            critic_carry,
        )
        critic_target_params = self.critic_network.init(
            critic_key,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            critic_carry,
        )
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        *_, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        transition = Transition(
            obs=obs,
            prev_done=done,
            action=action,
            reward=reward,
            next_obs=obs,
            done=done,
            info=info,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        critic_hidden_state = self.critic_network.initialize_carry(obs.shape)

        return key, SACState(
            step=0,
            timestep=timestep.from_sequence(),
            actor_hidden_state=actor_hidden_state,
            critic_hidden_state=critic_hidden_state,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
        )

    @partial(jax.jit, static_argnames=["self"])
    def _update_alpha(self, key, state: SACState):
        action_dim = self.env.action_space(self.env_params).shape[0]
        target_entropy = -self.cfg.target_entropy_scale * action_dim

        timestep = state.timestep
        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
        )

        key, sample_key = jax.random.split(key)
        actions = dist.sample(seed=sample_key)
        entropy = (-dist.log_prob(actions)).mean()

        def alpha_loss_fn(alpha_params):
            log_alpha = self.alpha_network.apply(alpha_params)
            alpha = jnp.exp(log_alpha)
            alpha_loss = alpha * (entropy - target_entropy).mean()
            return alpha_loss, {"alpha": alpha, "alpha_loss": alpha_loss}

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

    @partial(jax.jit, static_argnames=["self"])
    def _update_actor(
        self,
        key,
        state: SACState,
        batch,
        initial_actor_carry=None,
        initial_critic_carry=None,
    ):
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        def actor_loss_fn(actor_params):
            _, (dist, _) = self.actor_network.apply(
                actor_params,
                batch.obs,
                batch.prev_done,
                batch.action,
                add_feature_axis(batch.reward),
                batch.prev_done,
                initial_actor_carry,
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, ((q1, _), (q2, _)) = self.critic_network.apply(
                state.critic_params,
                batch.obs,
                batch.prev_done,
                actions,
                add_feature_axis(batch.reward),
                batch.prev_done,
                initial_critic_carry,
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * alpha - remove_feature_axis(q)).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        (_, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor_params
        )
        updates, actor_optimizer_state = self.actor_optimizer.update(
            grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, updates)

        state = state.replace(
            actor_params=actor_params, actor_optimizer_state=actor_optimizer_state
        )

        return state, info

    @partial(jax.jit, static_argnames=["self"])
    def _update_critic(
        self,
        key,
        state: SACState,
        batch,
        initial_actor_carry=None,
        initial_critic_carry=None,
        initial_critic_target_carry=None,
    ):
        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            batch.next_obs,
            batch.done,
            batch.action,
            add_feature_axis(batch.reward),
            batch.done,
            initial_actor_carry,
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, ((next_q1, _), (next_q2, _)) = self.critic_network.apply(
            state.critic_target_params,
            batch.next_obs,
            batch.done,
            next_actions,
            add_feature_axis(batch.reward),
            batch.done,
            initial_critic_target_carry,
        )
        next_q = jnp.minimum(next_q1, next_q2)

        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)
        target_q = batch.reward + self.cfg.gamma * (1 - batch.done) * (remove_feature_axis(next_q) - alpha * next_log_probs)

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            _, ((q1, aux1), (q2, aux2)) = self.critic_network.apply(
                critic_params,
                batch.obs,
                batch.prev_done,
                batch.action,
                add_feature_axis(batch.reward),
                batch.prev_done,
                initial_critic_carry,
            )
            critic_loss = (
                self.critic_network.head1.loss(q1, aux1, target_q)
                + self.critic_network.head2.loss(q2, aux2, target_q)
            )

            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
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
        batch = self.buffer.sample(state.buffer_state, batch_key).experience

        initial_actor_carry = None
        initial_critic_carry = None
        initial_critic_target_carry = None

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], batch
            )
            # Process burn-in through actor network
            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.prev_done,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)

            # Process burn-in through critic network
            initial_critic_carry, ((_, _), (_, _)) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.prev_done,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)

            # Process burn-in through target critic network
            initial_critic_target_carry, ((_, _), (_, _)) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_target_params),
                burn_in.next_obs,
                burn_in.done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.done,
            )
            initial_critic_target_carry = jax.lax.stop_gradient(initial_critic_target_carry)

            # Use remaining experience for actual learning
            batch = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], batch
            )

        state, critic_info = self._update_critic(
            critic_key,
            state,
            batch,
            initial_actor_carry,
            initial_critic_carry,
            initial_critic_target_carry,
        )
        state, actor_info = self._update_actor(
            actor_key,
            state,
            batch,
            initial_actor_carry,
            initial_critic_carry,
        )
        state, alpha_info = self._update_alpha(alpha_key, state)

        info = {**critic_info, **actor_info, **alpha_info}
        return state, info

    def _update_step(self, carry, _):
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            carry,
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, update_key = jax.random.split(key)
        state, update_info = self._update(update_key, state)

        transitions.info.update(update_info)

        return (key, state), transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key, state: SACState, num_steps: int) -> tuple[Key, SACState]:
        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: SACState, num_steps: int):
        (key, state), info = jax.lax.scan(
            self._update_step,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )

        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: SACState, num_steps: int):
        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, self.cfg.num_eval_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_eval_envs,) + action_space.shape, dtype=jnp.float32
        )
        reward = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.float32)
        done = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        hidden_state = self.actor_network.initialize_carry(obs.shape)

        critic_hidden_state = self.critic_network.initialize_carry(obs.shape)
        eval_state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_hidden_state=hidden_state,
            critic_hidden_state=critic_hidden_state,
        )

        (key, eval_state), transitions = jax.lax.scan(
            partial(
                self._step,
                policy=self._deterministic_action,
                write_to_buffer=False,
            ),
            (key, eval_state),
            length=num_steps // self.cfg.num_eval_envs,
        )

        return key, transitions
