from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.utils import Transition, periodic_incremental_update
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
)


@struct.dataclass
class Batch:
    """Data structure for a batch of transitions sampled from the replay buffer."""

    obs: Array
    """Batch of obs with shape [batch_size, obs_dim]"""
    action: Array
    """Batch of actions with shape [batch_size, action_dim]"""
    reward: Array
    """Batch of rewards with shape [batch_size]"""
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

    @property
    def target_entropy_multiplier(self) -> float:
        return self.target_entropy_scale


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
        dist = self.actor_network.apply(state.actor_params, state.obs, temperature=0.0)
        action = dist.sample(seed=sample_key)
        return key, action

    def _stochastic_action(self, key, state: SACState):
        key, sample_key = jax.random.split(key)
        dist = self.actor_network.apply(state.actor_params, state.obs)
        action = dist.sample(seed=sample_key)
        return key, action

    def _random_action(self, key, state: SACState):
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action

    def _step(self, carry, _, *, policy: Callable, write_to_buffer: bool = True):
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

    @partial(jax.jit, static_argnames=["self"])
    def _update_alpha(self, key, state: SACState):
        action_dim = self.env.action_space(self.env_params).shape[0]
        target_entropy = -self.cfg.target_entropy_multiplier * action_dim

        dist = self.actor_network.apply(state.actor_params, state.obs)

        key, sample_key = jax.random.split(key)
        actions = dist.sample(seed=sample_key)
        entropy = (-dist.log_prob(actions)).mean()

        def alpha_loss_fn(alpha_params):
            log_alpha = self.alpha_network.apply(alpha_params)
            alpha = jnp.exp(log_alpha)

            alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy)
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
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        actions = batch.first.action
        q1, q2 = self.critic_network.apply(
            state.critic_params, batch.first.obs, actions
        )
        q = jnp.minimum(q1, q2)

        def actor_loss_fn(actor_params):
            dist = self.actor_network.apply(actor_params, batch.first.obs)
            _, log_probs = dist.sample_and_log_prob(seed=key)
            actor_loss = (log_probs * alpha - q).mean()
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
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        dist = self.actor_network.apply(state.actor_params, batch.second.obs)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        next_q1, next_q2 = self.critic_network.apply(
            state.critic_target_params, batch.second.obs, next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch.first.reward + self.cfg.gamma * (1 - batch.first.done) * next_q

        def critic_loss_fn(critic_params):
            q1_values, q2_values = self.critic_network.apply(
                critic_params, batch.first.obs, batch.first.action
            )
            q1_loss = optax.huber_loss(q1_values, target_q).mean()
            q2_loss = optax.huber_loss(q2_values, target_q).mean()
            critic_loss = q1_loss + q2_loss

            return critic_loss, {
                "losses/q1_loss": q1_loss,
                "losses/q2_loss": q2_loss,
                "losses/critic_loss": critic_loss,
                "losses/q1_values": q1_values.mean(),
                "losses/q2_values": q2_values.mean(),
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

    def _update(self, key, state):
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

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (
                self.cfg.num_envs,
                self.env.action_space(self.env_params).shape[0],
            )
        )
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        actor_params = self.actor_network.init(actor_key, obs)
        actor_optimizer_state = self.actor_optimizer.init(actor_params)

        critic_params = self.critic_network.init(critic_key, obs, action)
        critic_target_params = self.critic_network.init(critic_key, obs, action)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        transition = Transition(obs=obs, action=action, reward=reward, done=done, info=info)  # type: ignore
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return key, SACState(
            step=0,
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
        )

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

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        state = state.replace(obs=obs, env_state=env_state)

        (key, state), transitions = jax.lax.scan(
            partial(
                self._step, policy=self._deterministic_action, write_to_buffer=False
            ),
            (key, state),
            length=num_steps,
        )

        return key, transitions
