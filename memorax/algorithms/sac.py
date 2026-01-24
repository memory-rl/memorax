from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.utils import Transition, periodic_incremental_update
from memorax.utils.typing import (Array, Buffer, BufferState, Environment,
                                  EnvParams, EnvState, Key)


@struct.dataclass
class Batch:
    """Data structure for a batch of transitions sampled from the replay buffer."""

    obs: Array
    """Batch of obs with shape [batch_size, obs_dim]"""
    prev_done: Array
    """Batch of prev done flags with shape [batch_size]"""
    action: Array
    """Batch of actions with shape [batch_size, action_dim]"""
    reward: Array
    """Batch of rewards with shape [batch_size]"""
    next_obs: Array
    """Batch of next obs with shape [batch_size, obs_dim]"""
    done: Array
    """Batch of done flags with shape [batch_size]"""


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
    mask: bool


@struct.dataclass(frozen=True)
class SACState:
    step: int
    env_state: EnvState
    buffer_state: BufferState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    critic_target_params: core.FrozenDict[str, Any]
    alpha_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    obs: Array
    done: Array
    actor_hidden_state: Array


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
        next_hidden_state, dist = self.actor_network.apply(
            state.actor_params,
            jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.actor_hidden_state,
            temperature=0.0,
        )
        action = dist.sample(seed=sample_key).squeeze(1)
        return key, (action, next_hidden_state)

    def _stochastic_action(self, key, state: SACState):
        key, sample_key = jax.random.split(key)
        next_hidden_state, dist = self.actor_network.apply(
            state.actor_params,
            jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.actor_hidden_state,
        )
        action = dist.sample(seed=sample_key).squeeze(1)
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

        num_envs = state.obs.shape[0]
        env_keys = jax.random.split(env_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(env_keys, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,  # type: ignore
            prev_done=state.done,  # type: ignore
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
            obs=next_obs,
            done=done,
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
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        actor_carry = self.actor_network.initialize_carry(obs.shape)
        actor_params = self.actor_network.init(
            actor_key,
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=actor_carry,
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        actor_hidden_state = self.actor_network.initialize_carry(obs.shape)

        critic_carry = self.critic_network.initialize_carry(obs.shape)
        critic_params = self.critic_network.init(
            critic_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            initial_carry=critic_carry,
            action=action,
        )
        critic_target_params = self.critic_network.init(
            critic_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            initial_carry=critic_carry,
            action=action,
        )
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

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

        return key, SACState(
            step=0,
            actor_hidden_state=actor_hidden_state,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            obs=obs,
            done=done,
        )

    @partial(jax.jit, static_argnames=["self"])
    def _update_alpha(self, key, state: SACState):
        action_dim = self.env.action_space(self.env_params).shape[0]
        target_entropy = -self.cfg.target_entropy_scale * action_dim

        _, dist = self.actor_network.apply(state.actor_params, state.obs, state.done)

        key, sample_key = jax.random.split(key)
        actions = dist.sample(seed=sample_key)
        entropy = (-dist.log_prob(actions)).mean()

        def alpha_loss_fn(alpha_params):
            alpha = self.alpha_network.apply(alpha_params)
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
    def _update_actor(self, key, state: SACState, batch: Batch):
        alpha = self.alpha_network.apply(state.alpha_params)
        mask = jnp.ones_like(batch.prev_done, dtype=bool)
        if self.cfg.mask:
            episode_idx = jnp.cumsum(batch.prev_done.astype(jnp.int32), axis=1)
            terminal = (episode_idx == 1) & batch.prev_done
            mask = (episode_idx == 0) | terminal

        def actor_loss_fn(actor_params):
            _, dist = self.actor_network.apply(actor_params, batch.obs, batch.prev_done)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, (q1, q2) = self.critic_network.apply(
                state.critic_params, batch.obs, batch.prev_done, action=actions
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * alpha - q.squeeze(-1)).mean(where=mask)
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
    def _update_critic(self, key, state: SACState, batch: Batch):
        _, dist = self.actor_network.apply(
            state.actor_params,
            batch.next_obs,
            batch.done,
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, (next_q1, next_q2) = self.critic_network.apply(
            state.critic_target_params, batch.next_obs, batch.done, action=next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)

        alpha = self.alpha_network.apply(state.alpha_params)
        target_q = batch.reward + self.cfg.gamma * (1 - batch.done) * next_q.squeeze(-1)

        target_q = jax.lax.stop_gradient(target_q)

        mask = jnp.ones_like(batch.prev_done, dtype=bool)
        if self.cfg.mask:
            episode_idx = jnp.cumsum(batch.prev_done.astype(jnp.int32), axis=1)
            terminal = (episode_idx == 1) & batch.prev_done
            mask = (episode_idx == 0) | terminal

        def critic_loss_fn(critic_params):
            _, (q1, q2) = self.critic_network.apply(
                critic_params, batch.obs, batch.prev_done, action=batch.action
            )
            q1_error = q1.squeeze(-1) - target_q
            q2_error = q2.squeeze(-1) - target_q
            critic_loss = (q1_error**2 + q2_error**2).mean(where=mask)

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

        state, critic_info = self._update_critic(critic_key, state, batch)
        state, actor_info = self._update_actor(actor_key, state, batch)
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

        eval_obs, eval_env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        eval_done = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.bool_)
        eval_hidden_state = self.actor_network.initialize_carry(eval_obs.shape)

        eval_state = state.replace(
            obs=eval_obs,
            done=eval_done,
            env_state=eval_env_state,
            actor_hidden_state=eval_hidden_state,
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
