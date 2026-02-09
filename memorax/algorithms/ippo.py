from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax import core
from flax import linen as nn
from flax import struct

from memorax.networks.sequence_models.utils import (add_feature_axis,
                                                    add_time_axis,
                                                    remove_feature_axis,
                                                    remove_time_axis)
from memorax.utils import (Timestep, Transition,
                           generalized_advantage_estimation)
from memorax.utils.typing import Array, Key

to_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(add_time_axis)(x), timestep
)

from_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(remove_time_axis)(x), timestep
)


@struct.dataclass(frozen=True)
class IPPOConfig:
    name: str
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
    target_kl: Optional[float] = None
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class IPPOState:
    step: int
    timestep: Timestep
    env_state: Any
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Array


@struct.dataclass(frozen=True)
class IPPO:
    cfg: IPPOConfig
    env: Any
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    def _deterministic_action(
        self, key: Key, state: IPPOState
    ) -> tuple[Key, IPPOState, Array, Array, None]:
        timestep = to_sequence(state.timestep)

        actor_carry, (probs, _) = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_carry,
        )

        action = jnp.argmax(probs.logits, axis=-1)
        log_prob = probs.log_prob(action)

        action = jax.vmap(remove_time_axis)(action)
        log_prob = jax.vmap(remove_time_axis)(log_prob)

        state = state.replace(actor_carry=actor_carry)
        return key, state, action, log_prob, None

    def _stochastic_action(
        self, key: Key, state: IPPOState
    ) -> tuple[Key, IPPOState, Array, Array, Array]:
        key, action_key, actor_memory_key, critic_memory_key = jax.random.split(key, 4)
        timestep = to_sequence(state.timestep)

        actor_carry, (probs, _) = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_carry,
            rngs={"memory": actor_memory_key},
        )

        action_keys = jax.random.split(action_key, self.env.num_agents)
        action, log_prob = jax.vmap(lambda p, k: p.sample_and_log_prob(seed=k))(
            probs, action_keys
        )

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.critic_carry,
            rngs={"memory": critic_memory_key},
        )

        action = jax.vmap(remove_time_axis)(action)
        log_prob = jax.vmap(remove_time_axis)(log_prob)
        value = jax.vmap(remove_time_axis)(value)
        value = remove_feature_axis(value)

        state = state.replace(actor_carry=actor_carry, critic_carry=critic_carry)
        return key, state, action, log_prob, value

    def _step(self, carry: tuple, _, *, policy: Callable):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, log_prob, value = policy(action_key, state)

        _, num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 1), out_axes=(1, 0, 1, 1, 0)
        )(step_keys, state.env_state, action)

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
            done=done,
            info=info,
            log_prob=log_prob,
            value=value,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=state.timestep.done,
        )

        state = state.replace(
            step=state.step + num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            env_state=env_state,
        )
        return (key, state), transition

    def _update_actor(
        self, key, state: IPPOState, initial_actor_carry, transitions, advantages
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )

            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.prev_action,
                add_feature_axis(burn_in.prev_reward),
                burn_in.prev_done,
                initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, :, self.cfg.burn_in_length :]

        def actor_loss_fn(params):
            _, (probs, _) = self.actor_network.apply(
                params,
                transitions.obs,
                transitions.prev_done,
                transitions.prev_action,
                add_feature_axis(transitions.prev_reward),
                transitions.prev_done,
                initial_actor_carry,
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
                jnp.clip(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
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
        self, key, state: IPPOState, initial_critic_carry, transitions, returns
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )

            initial_critic_carry, (_, _) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.prev_action,
                add_feature_axis(burn_in.prev_reward),
                burn_in.prev_done,
                initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            returns = returns[:, :, self.cfg.burn_in_length :]

        def critic_loss_fn(params):
            _, (values, aux) = self.critic_network.apply(
                params,
                transitions.obs,
                transitions.prev_done,
                transitions.prev_action,
                add_feature_axis(transitions.prev_reward),
                transitions.prev_done,
                initial_critic_carry,
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
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return key, state, critic_loss.mean()

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
            num_agents = self.env.num_agents
            num_envs = self.cfg.num_envs
            num_steps = self.cfg.num_steps

            if shuffle_time_axis:


                batch = (
                    initial_actor_carry,
                    initial_critic_carry,
                    *jax.tree.map(
                        lambda x: x.reshape(num_agents, -1, 1, *x.shape[3:]),
                        (transitions, advantages, returns),
                    ),
                )
                num_samples_per_agent = num_envs * num_steps
            else:
                num_samples_per_agent = num_envs


            permutation = jax.random.permutation(permutation_key, num_samples_per_agent)

            minibatches = jax.tree.map(
                lambda x: (
                    jnp.moveaxis(
                        jnp.take(x, permutation, axis=1).reshape(
                            num_agents, self.cfg.num_minibatches, -1, *x.shape[2:]
                        ),
                        1, 0,
                    )
                    if x is not None
                    else None
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

        timestep = to_sequence(state.timestep)

        _, (value, _) = self.critic_network.apply(
            state.critic_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.critic_carry,
        )
        value = jax.vmap(remove_time_axis)(value)
        value = remove_feature_axis(value)

        advantages, returns = generalized_advantage_estimation(
            self.cfg.gamma,
            self.cfg.gae_lambda,
            value,
            transitions,
        )

        transitions = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, min(2, x.ndim - 1)), transitions
        )
        advantages = jnp.moveaxis(advantages, 0, 2)
        returns = jnp.moveaxis(returns, 0, 2)

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
            lambda x: jnp.expand_dims(x, axis=(0, 1, 2)), metrics
        )
        info = {
            **transitions.info,
            "losses/actor_loss": actor_loss,
            "losses/critic_loss": critic_loss,
            "losses/entropy": entropy,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac,
        }

        return (key, state), transitions.replace(obs=None, next_obs=None, info=info)

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
        ) = jax.random.split(key, 8)

        agent_ids = self.env.agents
        num_agents = self.env.num_agents

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, out_axes=(1, 0))(env_keys)

        action_space = self.env.action_spaces[agent_ids[0]]

        action = jnp.zeros(
            (num_agents, self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_envs), dtype=jnp.bool)

        timestep = to_sequence(
            Timestep(obs=obs, action=action, reward=reward, done=done)
        )

        actor_carry = self.actor_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )
        critic_carry = self.critic_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )

        actor_params = self.actor_network.init(
            {
                "params": actor_key,
                "memory": actor_memory_key,
                "dropout": actor_dropout_key,
            },
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            actor_carry,
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "memory": critic_memory_key,
                "dropout": critic_dropout_key,
            },
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            critic_carry,
        )

        return (
            key,
            IPPOState(
                step=0,
                timestep=from_sequence(timestep),
                env_state=env_state,
                actor_params=actor_params,
                critic_params=critic_params,
                actor_optimizer_state=self.actor_optimizer.init(actor_params),
                critic_optimizer_state=self.critic_optimizer.init(critic_params),
                actor_carry=actor_carry,
                critic_carry=critic_carry,
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
            lambda x: jnp.moveaxis(x, 3, 1).reshape(-1, x.shape[1], x.shape[2], *x.shape[4:]),
            transitions,
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps", "deterministic"])
    def evaluate(self, key, state, num_steps, deterministic=True):
        key, reset_key = jax.random.split(key)
        num_agents = self.env.num_agents

        reset_keys = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, out_axes=(1, 0))(reset_keys)

        action_space = self.env.action_spaces[self.env.agents[0]]
        action = jnp.zeros(
            (num_agents, self.cfg.num_eval_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_eval_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_eval_envs), dtype=jnp.bool)

        actor_carry = self.actor_network.initialize_carry(
            (num_agents, self.cfg.num_eval_envs, None)
        )
        critic_carry = self.critic_network.initialize_carry(
            (num_agents, self.cfg.num_eval_envs, None)
        )

        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            env_state=env_state,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
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
