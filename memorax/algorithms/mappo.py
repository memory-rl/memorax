from functools import partial
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.networks.sequence_models.utils import (
    add_feature_axis,
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import Timestep, Transition, memory_metrics
from memorax.utils.typing import Array, Discrete, Environment, EnvParams, EnvState, Key

to_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(add_time_axis)(x), timestep
)

from_sequence = lambda timestep: jax.tree.map(
    lambda x: jax.vmap(remove_time_axis)(x), timestep
)


@struct.dataclass(frozen=True)
class MAPPOConfig:
    num_envs: int
    num_steps: int
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    target_kl: Optional[float] = None
    burn_in_length: int = 0
    centralized_critic: bool = False

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class MAPPOState:
    step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Array


@struct.dataclass(frozen=True)
class MAPPO:
    cfg: MAPPOConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    @property
    def _env_vectorized(self) -> bool:
        return getattr(self.env, "vectorized", False)

    def _env_reset(self, keys: Array, num_envs: int):
        if self._env_vectorized:
            obs, env_state = self.env.reset(keys[0])
            return obs, env_state
        else:
            return jax.vmap(self.env.reset, out_axes=(1, 0))(keys)

    def _env_step(self, keys: Array, env_state, actions: Array):
        if self._env_vectorized:
            return self.env.step(keys[0], env_state, actions)
        else:
            return jax.vmap(self.env.step, in_axes=(0, 0, 1), out_axes=(1, 0, 1, 1, 0))(
                keys, env_state, actions
            )

    def _deterministic_action(
        self, key: Key, state: MAPPOState
    ) -> tuple[Key, MAPPOState, Array, Array, None, dict]:
        timestep = to_sequence(state.timestep)

        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            mutable=["intermediates"],
        )

        action = (
            jnp.argmax(probs.logits, axis=-1)
            if isinstance(self.env.action_spaces[self.env.agents[0]], Discrete)
            else probs.mode()
        )
        log_prob = probs.log_prob(action)

        action = jax.vmap(remove_time_axis)(action)
        log_prob = jax.vmap(remove_time_axis)(log_prob)

        state = state.replace(actor_carry=actor_carry)
        return key, state, action, log_prob, None, intermediates

    def _stochastic_action(
        self, key: Key, state: MAPPOState
    ) -> tuple[Key, MAPPOState, Array, Array, Array, dict]:
        key, action_key, actor_memory_key, critic_memory_key = jax.random.split(key, 4)
        timestep = to_sequence(state.timestep)

        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.actor_carry,
            rngs={"memory": actor_memory_key},
            mutable=["intermediates"],
        )

        action_keys = jax.random.split(action_key, self.env.num_agents)
        sampled_action, log_prob = jax.vmap(lambda p, k: p.sample_and_log_prob(seed=k))(
            probs, action_keys
        )

        if self.cfg.centralized_critic:
            obs = jnp.moveaxis(timestep.obs, 0, 2)
            done = timestep.done[0]
            prev_action = timestep.action[0]
            reward = timestep.reward[0]

            critic_carry, (value, _) = self.critic_network.apply(
                state.critic_params,
                observation=obs,
                mask=done,
                action=prev_action,
                reward=add_feature_axis(reward),
                done=done,
                initial_carry=state.critic_carry,
                rngs={"memory": critic_memory_key},
            )

            action = jax.vmap(remove_time_axis)(sampled_action)
            log_prob = jax.vmap(remove_time_axis)(log_prob)
            value = remove_time_axis(value)
            value = value.T
        else:
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

            action = jax.vmap(remove_time_axis)(sampled_action)
            log_prob = jax.vmap(remove_time_axis)(log_prob)
            value = jax.vmap(remove_time_axis)(value)
            value = remove_feature_axis(value)

        state = state.replace(actor_carry=actor_carry, critic_carry=critic_carry)
        return key, state, action, log_prob, value, intermediates

    def _generalized_advantage_estimation(self, carry, transition):
        advantage, next_value = carry
        delta = (
            self.critic_network.head.get_target(transition, next_value)
            - transition.value
        )
        advantage = (
            delta
            + self.critic_network.head.gamma
            * self.cfg.gae_lambda
            * (1 - transition.second.done)
            * advantage
        )
        return (advantage, transition.value), advantage

    def _step(self, carry: tuple, _, *, policy: Callable):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, log_prob, value, intermediates = policy(action_key, state)

        _, num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = self._env_step(
            step_keys, state.env_state, action
        )
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
            log_prob=log_prob,
            value=value,
        )

        state = state.replace(
            step=state.step + num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
        )
        return (key, state), transition

    def _update_actor(
        self, key, state: MAPPOState, initial_actor_carry, transitions, advantages
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )

            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                observation=burn_in.first.obs,
                mask=burn_in.first.done,
                action=burn_in.first.action,
                reward=add_feature_axis(burn_in.first.reward),
                done=burn_in.first.done,
                initial_carry=initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, :, self.cfg.burn_in_length :]

        def actor_loss_fn(params):
            _, (probs, _) = self.actor_network.apply(
                params,
                observation=transitions.first.obs,
                mask=transitions.first.done,
                action=transitions.first.action,
                reward=add_feature_axis(transitions.first.reward),
                done=transitions.first.done,
                initial_carry=initial_actor_carry,
                rngs={"memory": memory_key, "dropout": dropout_key},
            )

            log_probs = probs.log_prob(transitions.second.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.log_prob)
            approx_kl = jnp.mean(transitions.log_prob - log_probs)
            clipfrac = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )

            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()
            return actor_loss - self.cfg.entropy_coefficient * entropy, (
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
        self, key, state: MAPPOState, initial_critic_carry, transitions, returns
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.centralized_critic:
            if initial_critic_carry is not None:
                initial_critic_carry = jax.tree.map(
                    lambda x: x[0], initial_critic_carry
                )

            obs = jnp.moveaxis(transitions.first.obs, 0, 2)
            prev_done = transitions.first.done[0]
            prev_action = transitions.first.action[0]
            prev_reward = transitions.first.reward[0]

            if self.cfg.burn_in_length > 0:
                burn_in_obs = obs[:, : self.cfg.burn_in_length]
                burn_in_prev_done = prev_done[:, : self.cfg.burn_in_length]
                burn_in_prev_action = prev_action[:, : self.cfg.burn_in_length]
                burn_in_prev_reward = prev_reward[:, : self.cfg.burn_in_length]

                initial_critic_carry, (_, _) = self.critic_network.apply(
                    jax.lax.stop_gradient(state.critic_params),
                    observation=burn_in_obs,
                    mask=burn_in_prev_done,
                    action=burn_in_prev_action,
                    reward=add_feature_axis(burn_in_prev_reward),
                    done=burn_in_prev_done,
                    initial_carry=initial_critic_carry,
                )
                initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)

                obs = obs[:, self.cfg.burn_in_length :]
                prev_done = prev_done[:, self.cfg.burn_in_length :]
                prev_action = prev_action[:, self.cfg.burn_in_length :]
                prev_reward = prev_reward[:, self.cfg.burn_in_length :]
                returns = returns[:, self.cfg.burn_in_length :]

                transitions = jax.tree.map(
                    lambda x: x[:, :, self.cfg.burn_in_length :], transitions
                )

            returns_transformed = jnp.moveaxis(returns, 0, -1)

            def critic_loss_fn(params):
                _, (values, aux) = self.critic_network.apply(
                    params,
                    observation=obs,
                    mask=prev_done,
                    action=prev_action,
                    reward=add_feature_axis(prev_reward),
                    done=prev_done,
                    initial_carry=initial_critic_carry,
                    rngs={"memory": memory_key, "dropout": dropout_key},
                )

                critic_loss = self.critic_network.head.loss(
                    values, aux, returns_transformed, transitions=transitions
                )
                if self.cfg.clip_value_loss:
                    old_values = jnp.moveaxis(transitions.value, 0, -1)
                    clipped_value = old_values + jnp.clip(
                        (values - old_values),
                        -self.cfg.clip_coefficient,
                        self.cfg.clip_coefficient,
                    )
                    clipped_critic_loss = self.critic_network.head.loss(
                        clipped_value, aux, returns_transformed, transitions=transitions
                    )
                    critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
                critic_loss = critic_loss.mean()

                return critic_loss

        else:
            if self.cfg.burn_in_length > 0:
                burn_in = jax.tree.map(
                    lambda x: x[:, :, : self.cfg.burn_in_length], transitions
                )

                initial_critic_carry, (_, _) = self.critic_network.apply(
                    jax.lax.stop_gradient(state.critic_params),
                    observation=burn_in.first.obs,
                    mask=burn_in.first.done,
                    action=burn_in.first.action,
                    reward=add_feature_axis(burn_in.first.reward),
                    done=burn_in.first.done,
                    initial_carry=initial_critic_carry,
                )
                initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)
                transitions = jax.tree.map(
                    lambda x: x[:, :, self.cfg.burn_in_length :], transitions
                )
                returns = returns[:, :, self.cfg.burn_in_length :]

            def critic_loss_fn(params):
                _, (values, aux) = self.critic_network.apply(
                    params,
                    observation=transitions.first.obs,
                    mask=transitions.first.done,
                    action=transitions.first.action,
                    reward=add_feature_axis(transitions.first.reward),
                    done=transitions.first.done,
                    initial_carry=initial_critic_carry,
                    rngs={"memory": memory_key, "dropout": dropout_key},
                )
                values = remove_feature_axis(values)

                critic_loss = self.critic_network.head.loss(
                    values, aux, returns, transitions=transitions
                )
                if self.cfg.clip_value_loss:
                    clipped_value = transitions.value + jnp.clip(
                        (values - transitions.value),
                        -self.cfg.clip_coefficient,
                        self.cfg.clip_coefficient,
                    )
                    clipped_critic_loss = self.critic_network.head.loss(
                        clipped_value, aux, returns, transitions=transitions
                    )
                    critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
                critic_loss = critic_loss.mean()

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
                        1,
                        0,
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

        if self.cfg.centralized_critic and initial_critic_carry is not None:
            initial_critic_carry = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None], (self.env.num_agents, *x.shape)),
                initial_critic_carry,
            )

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            (key, state),
            length=self.cfg.num_steps,
        )

        timestep = to_sequence(state.timestep)

        if self.cfg.centralized_critic:
            obs = jnp.moveaxis(timestep.obs, 0, 2)
            done = timestep.done[0]
            action = timestep.action[0]
            reward = timestep.reward[0]

            _, (value, _) = self.critic_network.apply(
                state.critic_params,
                observation=obs,
                mask=done,
                action=action,
                reward=add_feature_axis(reward),
                done=done,
                initial_carry=state.critic_carry,
            )
            value = remove_time_axis(value)
            value = value.T
        else:
            _, (value, _) = self.critic_network.apply(
                state.critic_params,
                observation=timestep.obs,
                mask=timestep.done,
                action=timestep.action,
                reward=add_feature_axis(timestep.reward),
                done=timestep.done,
                initial_carry=state.critic_carry,
            )
            value = jax.vmap(remove_time_axis)(value)
            value = remove_feature_axis(value)

        _, advantages = jax.lax.scan(
            self._generalized_advantage_estimation,
            (jnp.zeros_like(value), value),
            transitions,
            reverse=True,
            unroll=16,
        )
        returns = advantages + transitions.value

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
        memory = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=(0, 1, 2)),
            memory_metrics(state.actor_carry, initial_actor_carry),
        )
        metadata = {
            **transitions.metadata,
            "losses/actor_loss": actor_loss,
            "losses/critic_loss": critic_loss,
            "losses/entropy": entropy,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac,
            **memory,
        }

        return (key, state), transitions.replace(
            first=transitions.first.replace(obs=None),
            second=transitions.second.replace(obs=None),
            metadata=metadata,
        )

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
        obs, env_state = self._env_reset(env_keys, self.cfg.num_envs)

        action_space = self.env.action_spaces[agent_ids[0]]

        action = jnp.zeros(
            (num_agents, self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_envs), dtype=jnp.bool_)

        timestep = to_sequence(
            Timestep(obs=obs, action=action, reward=reward, done=done)
        )

        actor_carry = self.actor_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )

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

        if self.cfg.centralized_critic:
            critic_carry = self.critic_network.initialize_carry(
                (self.cfg.num_envs, None)
            )
            obs = jnp.moveaxis(timestep.obs, 0, 2)
            critic_params = self.critic_network.init(
                {
                    "params": critic_key,
                    "memory": critic_memory_key,
                    "dropout": critic_dropout_key,
                },
                observation=obs,
                mask=timestep.done[0],
                action=timestep.action[0],
                reward=add_feature_axis(timestep.reward[0]),
                done=timestep.done[0],
                initial_carry=critic_carry,
            )
        else:
            critic_carry = self.critic_network.initialize_carry(
                (num_agents, self.cfg.num_envs, None)
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

        return (
            key,
            MAPPOState(
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
    def warmup(
        self, key: Key, state: MAPPOState, num_steps: int
    ) -> tuple[Key, MAPPOState]:
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: MAPPOState, num_steps: int):
        (key, state), transitions = jax.lax.scan(
            self._update_step,
            (key, state),
            length=num_steps // (self.cfg.num_envs * self.cfg.num_steps),
        )
        transitions = jax.tree.map(
            lambda x: jnp.moveaxis(x, 3, 1).reshape(
                -1, x.shape[1], x.shape[2], *x.shape[4:]
            ),
            transitions,
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps", "deterministic"])
    def evaluate(self, key: Key, state: MAPPOState, num_steps: int, deterministic=True):
        key, reset_key = jax.random.split(key)
        num_agents = self.env.num_agents

        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = self._env_reset(reset_keys, self.cfg.num_envs)

        action_space = self.env.action_spaces[self.env.agents[0]]
        action = jnp.zeros(
            (num_agents, self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        reward = jnp.zeros((num_agents, self.cfg.num_envs), dtype=jnp.float32)
        done = jnp.ones((num_agents, self.cfg.num_envs), dtype=jnp.bool_)

        actor_carry = self.actor_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )
        if self.cfg.centralized_critic:
            critic_carry = self.critic_network.initialize_carry(
                (self.cfg.num_envs, None)
            )
        else:
            critic_carry = self.critic_network.initialize_carry(
                (num_agents, self.cfg.num_envs, None)
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

        return key, transitions.replace(
            first=transitions.first.replace(obs=None),
            second=transitions.second.replace(obs=None),
        )
