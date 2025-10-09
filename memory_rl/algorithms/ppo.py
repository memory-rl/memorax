from typing import Any, Callable

from functools import partial

import chex
import jax
import jax.numpy as jnp
import optax
from flax import core

from memory_rl.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
)
from memory_rl.networks import Network
from memory_rl.utils import generalized_advantage_estimatation, Transition


@chex.dataclass(frozen=True)
class PPOConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    num_steps: int
    anneal_lr: bool
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    learning_starts: int
    actor: Any
    critic: Any

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@chex.dataclass(frozen=True)
class PPOState:
    step: int
    obs: Array
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class PPO:
    cfg: PPOConfig
    env: Environment
    env_params: EnvParams
    actor: Network
    critic: Network
    optimizer: optax.GradientTransformation

    def _deterministic_action(
        self, key: Key, state: PPOState
    ) -> tuple[Key, Array, Array, Array]:
        probs = self.actor.apply(state.actor_params, state.obs)
        action = jnp.argmax(probs.logits, axis=-1)
        log_prob = probs.log_prob(action)
        value = self.critic.apply(state.critic_params, state.obs).squeeze(-1)
        return key, action, log_prob, value

    def _stochastic_action(
        self, key: Key, state: PPOState
    ) -> tuple[Key, Array, Array, Array]:
        key, action_key = jax.random.split(key)
        probs = self.actor.apply(state.actor_params, state.obs)
        action = probs.sample(seed=action_key)
        log_prob = probs.log_prob(action)
        value = self.critic.apply(state.critic_params, state.obs).squeeze(-1)
        return key, action, log_prob, value

    def _step(self, carry: tuple, _, *, policy: Callable):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action, log_prob, value = policy(action_key, state)

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
            log_prob=log_prob,  # type: ignore
            value=value,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,
            env_state=env_state,
        )
        return (key, state), transition

    def _update_actor(self, state: PPOState, transitions, advantages):

        def actor_loss_fn(params):
            probs = self.actor.apply(params, transitions.obs)
            log_prob = probs.log_prob(transitions.action)
            entropy = probs.entropy().mean()

            ratio = jnp.exp(log_prob - transitions.log_prob)
            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                )
                * advantages,
            ).mean()
            return actor_loss - self.cfg.ent_coef * entropy, entropy.mean()

        (actor_loss, entropy), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        actor_updates, actor_optimizer_state = self.optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
        )

        return state, actor_loss.mean(), entropy

    def _update_critic(self, state: PPOState, transitions, returns):

        def critic_loss_fn(params):
            value = self.critic.apply(params, transitions.obs).squeeze(-1)

            if self.cfg.clip_vloss:
                critic_loss = jnp.square(value - returns)
                clipped_value = transitions.value + jnp.clip(
                    (value - transitions.value),
                    -self.cfg.clip_coef,
                    self.cfg.clip_coef,
                )
                clipped_critic_loss = jnp.square(clipped_value - returns)
                critic_loss = 0.5 * jnp.maximum(critic_loss, clipped_critic_loss).mean()
            else:
                critic_loss = 0.5 * jnp.square(value - returns).mean()

            return self.cfg.vf_coef * critic_loss

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            state.critic_params
        )
        critic_updates, critic_optimizer_state = self.optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        state = state.replace(
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
        )
        return state, critic_loss.mean()

    def _update_minibatch(self, state, minibatch: tuple):
        transitions, advantages, returns = minibatch

        state, critic_loss = self._update_critic(state, transitions, returns)
        state, actor_loss, entropy = self._update_actor(state, transitions, advantages)

        return state, (actor_loss.mean(), critic_loss.mean(), entropy)

    def _update_epoch(self, carry: tuple, _):
        key, state, batch = carry

        key, permutation_key = jax.random.split(key)

        permutation = jax.random.permutation(permutation_key, self.cfg.batch_size)
        flattened_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        shuffled_batch = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=0), flattened_batch
        )
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x, [self.cfg.num_minibatches, -1] + list(x.shape[1:])
            ),
            shuffled_batch,
        )

        state, (actor_loss, critic_loss, entropy) = jax.lax.scan(
            self._update_minibatch,
            state,
            minibatches,
        )

        return (
            key,
            state,
            batch,
        ), (actor_loss, critic_loss, entropy)

    def _update_step(self, carry: tuple, _):

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            carry,
            length=self.cfg.num_steps,
        )
        final_value = self.critic.apply(state.critic_params, state.obs).squeeze(-1)

        advantages, returns = generalized_advantage_estimatation(
            self.cfg.gamma,
            self.cfg.gae_lambda,
            final_value,
            transitions,
        )

        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = (transitions, advantages, returns)

        (key, state, batch), (actor_loss, critic_loss, entropy) = jax.lax.scan(
            self._update_epoch,
            (key, state, batch),
            length=self.cfg.update_epochs,
        )
        transitions, *_ = batch

        transitions.info["losses/actor_loss"] = actor_loss
        transitions.info["losses/critic_loss"] = critic_loss
        transitions.info["losses/entropy"] = entropy

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key):
        key, env_key, actor_key, critic_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )

        actor_params = self.actor.init(actor_key, obs)
        actor_optimizer_state = self.optimizer.init(actor_params)

        critic_params = self.critic.init(critic_key, obs)
        critic_optimizer_state = self.optimizer.init(critic_params)

        return (
            key,
            PPOState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                actor_params=actor_params,  # type: ignore
                actor_optimizer_state=actor_optimizer_state,  # type: ignore
                critic_params=critic_params,  # type: ignore
                critic_optimizer_state=critic_optimizer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key", "state"])
    def warmup(self, key, state, num_steps):
        """No warmup needed for PPO"""
        return key, state

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def train(self, key, state, num_steps):

        (key, state), transitions = jax.lax.scan(
            self._update_step,
            (key, state),
            length=num_steps // self.cfg.num_envs // self.cfg.num_steps,
        )

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, -1, 1), transitions)
        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def evaluate(self, key, state, num_steps):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        state = state.replace(obs=obs, env_state=env_state)

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
