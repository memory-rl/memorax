from functools import partial
from typing import Any, Callable

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core

from memory_rl.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
)
from memory_rl.utils import periodic_incremental_update, Transition


@struct.dataclass
class Batch:
    """Data structure for a batch of transitions sampled from the replay buffer."""

    obs: Array
    """Batch of obs with shape [batch_size, obs_dim]"""
    action: Array
    """Batch of actions with shape [batch_size, action_dim]"""
    reward: Array
    """Batch of rewards with shape [batch_size]"""
    next_obs: Array
    """Batch of next obs with shape [batch_size, obs_dim]"""
    done: Array
    """Batch of next done flags with shape [batch_size]"""


@struct.dataclass(frozen=True)
class RSACDConfig:
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
    sequence_length: int
    burn_in_length: int
    mask: bool


@struct.dataclass(frozen=True)
class RSACDState:
    step: int
    env_state: EnvState
    buffer_state: BufferState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    critic_target_params: core.FrozenDict[str, Any]
    alpha_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    actor_hidden_state: Any
    critic_hidden_state: Any
    alpha_optimizer_state: optax.OptState
    obs: Array
    done: Array


@struct.dataclass(frozen=True)
class RSACD:
    cfg: RSACDConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    alpha_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    buffer: Buffer

    def _deterministic_action(
        self, key: Key, state: RSACDState
    ) -> tuple[Key, RSACDState]:
        actor_hidden_state, dist = self.actor_network.apply(
            state.actor_params,
            observation=jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.actor_hidden_state,
        )
        action = dist.mode().squeeze(1)

        state = state.replace(
            actor_hidden_state=actor_hidden_state,
        )
        return key, state, action

    def _stochastic_action(
        self, key: Key, state: RSACDState
    ) -> tuple[Key, RSACDState, Array, Array]:
        key, sample_key = jax.random.split(key)
        actor_hidden_state, dist = self.actor_network.apply(
            state.actor_params,
            jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.actor_hidden_state,
        )
        action = dist.sample(seed=sample_key).squeeze(1)
        state = state.replace(
            actor_hidden_state=actor_hidden_state,
        )
        return key, state, action

    def _random_action(
        self, key: Key, state: RSACDState
    ) -> tuple[Key, RSACDState, Array, Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action

    def _step(
        self, carry, _, *, policy: Callable, write_to_buffer: bool = True
    ) -> tuple[Key, RSACDState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action = policy(action_key, state)
        num_envs = state.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            next_obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(state.buffer_state, transition)
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            env_state=env_state,  # type: ignore
            buffer_state=buffer_state,
        )
        return (key, state), transition

    @partial(jax.jit, static_argnames=["self"])
    def _update_alpha(self, state: RSACDState, batch: Batch):
        action_dim = self.env.action_space(self.env_params).n
        target_entropy = self.cfg.target_entropy_scale * jnp.log(action_dim)

        _, dist = self.actor_network.apply(
            state.actor_params, batch.obs, batch.done
        )
        entropy = dist.entropy().mean()

        def alpha_loss_fn(alpha_params):
            log_alpha = self.alpha_network.apply(alpha_params)
            alpha_loss = -log_alpha * jax.lax.stop_gradient(entropy - target_entropy)
            return alpha_loss, {
                "losses/log_alpha": log_alpha,
                "losses/alpha_loss": alpha_loss,
                "losses/entropy": entropy,
                "losses/target_entropy": target_entropy,
            }

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
    def _update_actor(self, key, state: RSACDState, batch: Batch):
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        mask = jnp.ones_like(batch.reward)
        if self.cfg.mask:
            episode_idx = jnp.cumsum(batch.done, axis=1)
            terminal = (episode_idx == 1) & batch.done
            mask = (episode_idx == 0) | terminal

        _, (q1, q2) = self.critic_network.apply(
            state.critic_params, batch.obs, batch.done
        )
        q = jnp.minimum(q1, q2)

        def actor_loss_fn(actor_params):
            _, dist = self.actor_network.apply(actor_params, batch.obs, batch.done)

            actor_loss = -(
                jnp.sum(dist.probs * q, axis=-1) + alpha * dist.entropy()
            ).mean()
            return actor_loss, {
                "losses/actor_loss": actor_loss,
                "losses/entropy": dist.entropy().mean(),
            }

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
    def _update_critic(self, key, state: RSACDState, batch: Batch):
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        _, dist = self.actor_network.apply(
            state.actor_params, batch.next_obs, batch.done
        )

        _, (next_q1, next_q2) = self.critic_network.apply(
            state.critic_target_params, batch.next_obs, batch.done
        )

        next_q = jnp.minimum(next_q1, next_q2)
        soft_state_values = (
            jnp.sum(dist.probs * next_q, axis=-1) + alpha * dist.entropy()
        )
        target_q = batch.reward + self.cfg.gamma * (1 - batch.done) * soft_state_values

        mask = jnp.ones_like(batch.reward)
        if self.cfg.mask:
            episode_idx = jnp.cumsum(batch.done, axis=1)
            terminal = (episode_idx == 1) & batch.done
            mask = (episode_idx == 0) | terminal

        def critic_loss_fn(critic_params):
            _, (q1_values, q2_values) = self.critic_network.apply(
                critic_params, batch.obs, batch.done
            )

            action = jnp.expand_dims(batch.action, -1)
            q1_value = jnp.take_along_axis(q1_values, action, axis=-1).squeeze(-1)
            q2_value = jnp.take_along_axis(q2_values, action, axis=-1).squeeze(-1)

            q1_loss = optax.huber_loss(q1_value, target_q).mean()
            q2_loss = optax.huber_loss(q2_value, target_q).mean()
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

        # Update target network periodically
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
        # Sample from buffer
        key, batch_key = jax.random.split(key)
        batch = self.buffer.sample(state.buffer_state, batch_key)

        # Update critic
        key, critic_key = jax.random.split(key)
        state, critic_info = self._update_critic(
            critic_key, state, batch.experience
        )  ### Because fbx has weird way of storing transitions

        # Update actor using updated critic
        key, actor_key = jax.random.split(key)
        state, actor_info = self._update_actor(
            actor_key, state, batch.experience
        )  ### Because fbx has weird way of storing transitions

        # Update alpha using updated actor and critic
        state, alpha_info = self._update_alpha(state, batch.experience)

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

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        # Initialize environment
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        # Initialize actor
        actor_params = self.actor_network.init(
            actor_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        actor_hidden_state = self.actor_network.initialize_carry(obs.shape)

        # Initialize critic
        critic_params = self.critic_network.init(
            critic_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        critic_target_params = self.critic_network.init(
            critic_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        critic_hidden_state = self.critic_network.initialize_carry(obs.shape)

        # Initialize alpha
        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        transition = Transition(obs=obs, action=action, reward=reward, next_obs=obs, done=done, info=info)  # type: ignore
        transition = jax.tree.map(lambda x: x[0], transition)

        buffer_state = self.buffer.init(transition)

        return key, RSACDState(
            step=0,
            obs=obs,
            done=done,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            actor_hidden_state=actor_hidden_state,
            critic_hidden_state=critic_hidden_state,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key, state: RSACDState, num_steps: int) -> tuple[Key, RSACDState]:

        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: RSACDState, num_steps: int):

        (key, state), transitions = jax.lax.scan(
            self._update_step,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: RSACDState, num_steps: int):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=jnp.bool)
        actor_hidden_state = self.actor_network.initialize_carry(obs.shape)
        state = state.replace(
            obs=obs,
            done=done,
            actor_hidden_state=actor_hidden_state,
            env_state=env_state,
        )

        (key, _), transitions = jax.lax.scan(
            partial(
                self._step, policy=self._deterministic_action, write_to_buffer=False
            ),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
