"""Implementing PPO with intrinsic reward support"""

from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax import core
from flax import linen as nn
from flax import struct

from memorax.networks.sequence_models.utils import (
    add_feature_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import Timestep, Transition, generalized_advantage_estimation, memory_metrics
from memorax.utils.typing import Array, Discrete, Environment, EnvParams, EnvState, Key


@struct.dataclass(frozen=True)
class IRPPOConfig:
    num_envs: int
    num_eval_envs: int
    num_steps: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    intrinsic_reward_coef: float = 0.01
    target_kl: Optional[float] = None
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class IRPPOState:
    step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Array
    intrinsic_reward_params: core.FrozenDict[str, Any]
    intrinsic_reward_optimizer_state: optax.OptState


@struct.dataclass(frozen=True)
class IRPPO:
    """PPO with pluggable intrinsic reward modules.

    The intrinsic_reward_network must implement:
    - intrinsic_reward(params, obs, next_obs, action) -> rewards
    - loss(params, obs, next_obs, action) -> (loss, metrics)
    """

    cfg: IRPPOConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    intrinsic_reward_network: nn.Module
    intrinsic_reward_optimizer: optax.GradientTransformation

    def _deterministic_action(
        self, key: Key, state: IRPPOState
    ) -> tuple[Key, IRPPOState, Array, Array, None, dict]:
        timestep = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            mutable=['intermediates'],
        )

        action = (
            jnp.argmax(probs.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else probs.mode()
        )
        log_prob = probs.log_prob(action)

        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        state = state.replace(
            actor_carry=actor_carry,
        )
        return key, state, action, log_prob, None, intermediates

    def _stochastic_action(
        self, key: Key, state: IRPPOState
    ) -> tuple[Key, IRPPOState, Array, Array, Array, dict]:
        (
            key,
            action_key,
            actor_memory_key,
            critic_memory_key,
        ) = jax.random.split(key, 4)

        timestep = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            rngs={"memory": actor_memory_key},
            mutable=['intermediates'],
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.critic_carry,
            rngs={"memory": critic_memory_key},
        )

        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        state = state.replace(
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )
        return key, state, action, log_prob, value, intermediates

    def _step(self, carry: tuple, _, *, policy: Callable):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, log_prob, value, intermediates = policy(action_key, state)

        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates_metrics = jax.tree.map(
            jnp.mean, intermediates.get('intermediates', {}),
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        prev_action = jnp.where(
            jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        transition = Transition(
            obs=state.timestep.obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            info={**info, "intermediates": intermediates_metrics},
            log_prob=log_prob,
            value=value,
            prev_action=prev_action,
            prev_reward=jnp.where(state.timestep.done, 0, state.timestep.reward),
            prev_done=state.timestep.done,
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=jnp.asarray(reward, dtype=jnp.float32), done=done),
            env_state=env_state,
        )
        return (key, state), transition

    def _update_actor(
        self, key, state: IRPPOState, initial_actor_carry, transitions, advantages
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
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
            transitions = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, self.cfg.burn_in_length :]

        def actor_loss_fn(params):
            _, (probs, _) = self.actor_network.apply(
                params,
                observation=transitions.obs,
                mask=transitions.prev_done,
                action=transitions.prev_action,
                reward=add_feature_axis(transitions.prev_reward),
                done=transitions.prev_done,
                initial_carry=initial_actor_carry,
                rngs={"memory": memory_key, "dropout": dropout_key},
            )
            log_probs = probs.log_prob(transitions.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.log_prob)
            approx_kl = jnp.mean(transitions.log_prob - log_probs)
            clipfrac = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coef).astype(jnp.float32)
            )

            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                )
                * advantages,
            ).mean()
            return actor_loss - self.cfg.ent_coef * entropy, (
                entropy.mean(),
                approx_kl.mean(),
                clipfrac.mean(),
            )

        (actor_loss, aux), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
        )
        return key, state, actor_loss.mean(), aux

    def _update_critic(
        self, key, state: IRPPOState, initial_critic_carry, transitions, returns
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
            )
            initial_critic_carry, (_, _) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                observation=burn_in.obs,
                mask=burn_in.prev_done,
                action=burn_in.prev_action,
                reward=add_feature_axis(burn_in.prev_reward),
                done=burn_in.prev_done,
                initial_carry=initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)
            transitions = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            returns = returns[:, self.cfg.burn_in_length :]

        def critic_loss_fn(params):
            _, (values, aux) = self.critic_network.apply(
                params,
                observation=transitions.obs,
                mask=transitions.prev_done,
                action=transitions.prev_action,
                reward=add_feature_axis(transitions.prev_reward),
                done=transitions.prev_done,
                initial_carry=initial_critic_carry,
                rngs={"memory": memory_key, "dropout": dropout_key},
            )
            values = remove_feature_axis(values)

            if self.cfg.clip_vloss:
                critic_loss = jnp.square(values - returns)
                clipped_value = transitions.value + jnp.clip(
                    (values - transitions.value),
                    -self.cfg.clip_coef,
                    self.cfg.clip_coef,
                )
                clipped_critic_loss = jnp.square(clipped_value - returns)
                critic_loss = 0.5 * jnp.maximum(critic_loss, clipped_critic_loss).mean()
            else:
                critic_loss = self.critic_network.head.loss(values, aux, returns)

            return critic_loss

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            state.critic_params
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        state = state.replace(
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
        )
        return key, state, critic_loss.mean()

    def _update_intrinsic_reward(
        self, state: IRPPOState, obs: Array, next_obs: Array, action: Array
    ):
        """Update intrinsic reward network."""

        def ir_loss_fn(params):
            return self.intrinsic_reward_network.loss(params, obs, next_obs, action)

        (loss, metrics), grads = jax.value_and_grad(ir_loss_fn, has_aux=True)(
            state.intrinsic_reward_params
        )
        updates, ir_optimizer_state = self.intrinsic_reward_optimizer.update(
            grads, state.intrinsic_reward_optimizer_state, state.intrinsic_reward_params
        )
        ir_params = optax.apply_updates(state.intrinsic_reward_params, updates)

        state = state.replace(
            intrinsic_reward_params=ir_params,
            intrinsic_reward_optimizer_state=ir_optimizer_state,
        )
        return state, metrics

    def _update_minibatch(self, carry, minibatch: tuple):
        key, state = carry
        (
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
        ) = minibatch

        key, state, critic_loss = self._update_critic(
            key, state, initial_critic_carry, transitions, returns
        )
        key, state, actor_loss, aux = self._update_actor(
            key, state, initial_actor_carry, transitions, advantages
        )

        return (key, state), (actor_loss, critic_loss, aux)

    def _update_epoch(self, carry: tuple):
        (
            key,
            state,
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
            *_,
            epoch,
        ) = carry

        key, permutation_key = jax.random.split(key)

        batch = (
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
        )

        def shuffle(batch):
            shuffle_time_axis = (
                initial_actor_carry is None or initial_critic_carry is None
            )
            num_permutations = self.cfg.num_envs
            if shuffle_time_axis:
                batch = (
                    initial_actor_carry,
                    initial_critic_carry,
                    *jax.tree.map(
                        lambda x: x.reshape(-1, 1, *x.shape[2:]),
                        (transitions, advantages, returns),
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

        (key, state), (actor_loss, critic_loss, (entropy, approx_kl, clipfrac)) = (
            jax.lax.scan(
                self._update_minibatch,
                (key, state),
                minibatches,
            )
        )

        metrics = jax.tree.map(
            lambda x: x.mean(), (actor_loss, critic_loss, entropy, approx_kl, clipfrac)
        )

        return (
            key,
            state,
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
            metrics,
            epoch + 1,
        )

    def _update_step(self, carry: tuple, _):
        key, state = carry
        initial_actor_carry = state.actor_carry
        initial_critic_carry = state.critic_carry
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            (key, state),
            length=self.cfg.num_steps,
        )

        obs_flat = transitions.obs.reshape(-1, *transitions.obs.shape[2:])
        next_obs_flat = transitions.next_obs.reshape(
            -1, *transitions.next_obs.shape[2:]
        )
        action_flat = transitions.action.reshape(-1)

        intrinsic_rewards = self.intrinsic_reward_network.intrinsic_reward(
            state.intrinsic_reward_params, obs_flat, next_obs_flat, action_flat
        )
        intrinsic_rewards = intrinsic_rewards.reshape(transitions.obs.shape[:2])

        state, ir_metrics = self._update_intrinsic_reward(
            state, obs_flat, next_obs_flat, action_flat
        )

        augmented_rewards = (
            transitions.reward + self.cfg.intrinsic_reward_coef * intrinsic_rewards
        )
        transitions = transitions.replace(reward=augmented_rewards)

        timestep = state.timestep.to_sequence()
        _, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.critic_carry,
        )
        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        advantages, returns = generalized_advantage_estimation(
            self.cfg.gamma,
            self.cfg.gae_lambda,
            value,
            transitions,
        )

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        advantages = jnp.swapaxes(advantages, 0, 1)
        returns = jnp.swapaxes(returns, 0, 1)

        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def cond_fun(carry):
            *_, (*_, approx_kl, _), epoch = carry

            cond = epoch < self.cfg.update_epochs

            if self.cfg.target_kl:
                cond = cond & (approx_kl < self.cfg.target_kl)

            return cond

        key, state, *_, metrics, _ = jax.lax.while_loop(
            cond_fun,
            self._update_epoch,
            (
                key,
                state,
                initial_actor_carry,
                initial_critic_carry,
                transitions,
                advantages,
                returns,
                (0.0, 0.0, 0.0, 0.0, 0.0),
                0,
            ),
        )

        actor_loss, critic_loss, entropy, approx_kl, clipfrac = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=(0, 1)), metrics
        )
        memory = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=(0, 1)),
            memory_metrics(state.actor_carry, initial_actor_carry),
        )
        info = {
            **transitions.info,
            "losses/actor_loss": actor_loss,
            "losses/critic_loss": critic_loss,
            "losses/entropy": entropy,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac,
            **{k: jnp.expand_dims(v, axis=(0, 1)) for k, v in ir_metrics.items()},
            **memory,
        }

        return (
            key,
            state,
        ), transitions.replace(obs=None, next_obs=None, info=info)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        (
            key,
            env_key,
            actor_key,
            actor_memory_key,
            actor_dropout_key,
            critic_key,
            critic_memory_key,
            critic_dropout_key,
            ir_key,
        ) = jax.random.split(key, 9)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))

        actor_params = self.actor_network.init(
            {
                "params": actor_key,
                "memory": actor_memory_key,
                "dropout": actor_dropout_key,
            },
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "memory": critic_memory_key,
                "dropout": critic_dropout_key,
            },
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=critic_carry,
        )

        obs_shape = self.env.observation_space(self.env_params).shape
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)
        ir_params = self.intrinsic_reward_network.init(
            ir_key, dummy_obs, dummy_obs, dummy_action
        )["params"]

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        ir_optimizer_state = self.intrinsic_reward_optimizer.init(ir_params)

        return (
            key,
            IRPPOState(
                step=0,
                timestep=timestep.from_sequence(),
                actor_carry=actor_carry,
                critic_carry=critic_carry,
                env_state=env_state,
                actor_params=actor_params,
                critic_params=critic_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
                intrinsic_reward_params=ir_params,
                intrinsic_reward_optimizer_state=ir_optimizer_state,
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key, state, num_steps):
        return key, state

    @partial(jax.jit, static_argnums=(0, 3))
    def train(self, key, state, num_steps):
        (key, state), transitions = jax.lax.scan(
            self._update_step,
            (key, state),
            length=num_steps // (self.cfg.num_envs * self.cfg.num_steps),
        )

        transitions = jax.tree.map(
            lambda x: (y := x.swapaxes(1, 2)).reshape((-1,) + y.shape[2:]),
            transitions,
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps", "deterministic"])
    def evaluate(self, key, state, num_steps, deterministic=True):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_eval_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros(self.cfg.num_eval_envs, dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_eval_envs, dtype=jnp.bool)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        initial_actor_carry = self.actor_network.initialize_carry(
            (self.cfg.num_eval_envs, None)
        )
        initial_critic_carry = self.critic_network.initialize_carry(
            (self.cfg.num_eval_envs, None)
        )
        state = state.replace(
            timestep=timestep,
            actor_carry=initial_actor_carry,
            critic_carry=initial_critic_carry,
            env_state=env_state,
        )

        policy = (
            self._deterministic_action if deterministic else self._stochastic_action
        )
        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=policy),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
