from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class PPOConfig:
    num_envs: int
    num_steps: int
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    target_kl: float | None = None
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PPOState:
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
class PPO:
    cfg: PPOConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    def _deterministic_action(
        self, key: Key, state: PPOState
    ) -> tuple[PPOState, Array, Array, None, dict]:
        timestep = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.actor_carry,
            mutable=["intermediates"],
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
        return state, action, log_prob, None, intermediates

    def _stochastic_action(
        self, key: Key, state: PPOState
    ) -> tuple[PPOState, Array, Array, Array, dict]:
        action_key, actor_torso_key, critic_torso_key = jax.random.split(key, 3)

        timestep = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.actor_carry,
            rngs={"torso": actor_torso_key},
            mutable=["intermediates"],
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.critic_carry,
            rngs={"torso": critic_torso_key},
        )

        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        state = state.replace(
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )
        return state, action, log_prob, value, intermediates

    def _generalized_advantage_estimation(self, carry: tuple, transition: Transition):
        advantage, next_value = carry
        delta = (
            self.critic_network.head.get_target(transition, next_value)
            - transition.aux["value"]
        )
        advantage = (
            delta
            + self.critic_network.head.gamma
            * self.cfg.gae_lambda
            * (1 - transition.second.done)
            * advantage
        )
        return (advantage, transition.aux["value"]), advantage

    def _step(
        self, state: PPOState, key: Key, *, policy: Callable
    ) -> tuple[PPOState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value, intermediates = policy(action_key, state)

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
        first = Timestep(
            obs=state.timestep.obs,
            action=jnp.where(
                jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
                jnp.zeros_like(state.timestep.action),
                state.timestep.action,
            ),
            reward=jnp.where(state.timestep.done, 0, state.timestep.reward),
            done=state.timestep.done,
        )
        second = Timestep(
            obs=None,
            action=action,
            reward=reward,
            done=done,
        )
        lox.log({"info": info, "intermediates": intermediates})

        transition = Transition(
            first=first,
            second=second,
            aux={"log_prob": log_prob, "value": value},
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
        )
        return state, transition

    def _update_actor(
        self, key: Key, state: PPOState, initial_actor_carry: Carry, transitions: Transition, advantages: Array
    ) -> tuple[PPOState, Array, tuple[Array, Array, Array]]:
        torso_key, dropout_key = jax.random.split(key)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
            )
            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                observation=burn_in.first.obs,
                done=burn_in.first.done,
                action=burn_in.first.action,
                reward=add_feature_axis(burn_in.first.reward),
                initial_carry=initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)
            transitions = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, self.cfg.burn_in_length :]

        def actor_loss_fn(params: PyTree):
            _, (probs, _) = self.actor_network.apply(
                params,
                observation=transitions.first.obs,
                done=transitions.first.done,
                action=transitions.first.action,
                reward=add_feature_axis(transitions.first.reward),
                initial_carry=initial_actor_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            log_probs = probs.log_prob(transitions.second.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
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
                approximate_kl.mean(),
                clip_fraction.mean(),
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
        return state, actor_loss.mean(), aux

    def _update_critic(
        self, key: Key, state: PPOState, initial_critic_carry: Carry, transitions: Transition, returns: Array
    ) -> tuple[PPOState, Array]:
        torso_key, dropout_key = jax.random.split(key)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
            )
            initial_critic_carry, (_, _) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                observation=burn_in.first.obs,
                done=burn_in.first.done,
                action=burn_in.first.action,
                reward=add_feature_axis(burn_in.first.reward),
                initial_carry=initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)
            transitions = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            returns = returns[:, self.cfg.burn_in_length :]

        def critic_loss_fn(params: PyTree):
            _, (values, aux) = self.critic_network.apply(
                params,
                observation=transitions.first.obs,
                done=transitions.first.done,
                action=transitions.first.action,
                reward=add_feature_axis(transitions.first.reward),
                initial_carry=initial_critic_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            values = remove_feature_axis(values)

            critic_loss = self.critic_network.head.loss(
                values, aux, returns, transitions=transitions
            )
            if self.cfg.clip_value_loss:
                clipped_value = transitions.aux["value"] + jnp.clip(
                    (values - transitions.aux["value"]),
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
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
        )
        return state, critic_loss.mean()

    def _update_minibatch(
        self, state: PPOState, xs: tuple
    ) -> tuple[PPOState, tuple[Array, Array, tuple[Array, Array, Array]]]:
        minibatch, key = xs
        (
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
        ) = minibatch

        actor_key, critic_key = jax.random.split(key)

        state, critic_loss = self._update_critic(
            critic_key, state, initial_critic_carry, transitions, returns
        )
        state, actor_loss, aux = self._update_actor(
            actor_key, state, initial_actor_carry, transitions, advantages
        )

        return state, (actor_loss, critic_loss, aux)

    def _update_epoch(self, carry: tuple) -> tuple:
        (
            state,
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
            *_,
            base_key,
            epoch,
        ) = carry

        key = jax.random.fold_in(base_key, epoch)
        permutation_key, minibatch_key = jax.random.split(key)

        batch = (
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
        )

        def shuffle(batch: PyTree):
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
        minibatch_keys = jax.random.split(minibatch_key, self.cfg.num_minibatches)

        state, (
            actor_loss,
            critic_loss,
            (entropy, approximate_kl, clip_fraction),
        ) = jax.lax.scan(
            self._update_minibatch,
            state,
            (minibatches, minibatch_keys),
        )

        metrics = jax.tree.map(
            lambda x: x.mean(),
            (actor_loss, critic_loss, entropy, approximate_kl, clip_fraction),
        )

        return (
            state,
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
            metrics,
            base_key,
            epoch + 1,
        )

    def _update_step(self, state: PPOState, key: Key) -> tuple[PPOState, None]:
        step_key, epoch_key = jax.random.split(key)

        initial_actor_carry = state.actor_carry
        initial_critic_carry = state.critic_carry

        step_keys = jax.random.split(step_key, self.cfg.num_steps)
        state, transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            state,
            step_keys,
        )

        timestep = state.timestep.to_sequence()
        _, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=state.critic_carry,
        )
        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        _, advantages = jax.lax.scan(
            self._generalized_advantage_estimation,
            (jnp.zeros_like(value), value),
            transitions,
            reverse=True,
            unroll=16,
        )
        returns = advantages + transitions.aux["value"]

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        advantages = jnp.swapaxes(advantages, 0, 1)
        returns = jnp.swapaxes(returns, 0, 1)

        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def cond_fun(carry):
            *_, (*_, approximate_kl, _), _base_key, epoch = carry

            cond = epoch < self.cfg.update_epochs

            if self.cfg.target_kl:
                cond = cond & (approximate_kl < self.cfg.target_kl)

            return cond

        state, *_, metrics, _base_key, _ = jax.lax.while_loop(
            cond_fun,
            self._update_epoch,
            (
                state,
                initial_actor_carry,
                initial_critic_carry,
                transitions,
                advantages,
                returns,
                (0.0, 0.0, 0.0, 0.0, 0.0),
                epoch_key,
                0,
            ),
        )

        actor_loss, critic_loss, entropy, approximate_kl, clip_fraction = metrics
        lox.log(
            {
                "losses/actor_loss": actor_loss,
                "losses/critic_loss": critic_loss,
                "losses/entropy": entropy,
                "losses/approximate_kl": approximate_kl,
                "losses/clip_fraction": clip_fraction,
            }
        )

        return state, None

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key: Key) -> PPOState:
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
        ) = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))

        actor_params = self.actor_network.init(
            {
                "params": actor_key,
                "torso": actor_torso_key,
                "dropout": actor_dropout_key,
            },
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "torso": critic_torso_key,
                "dropout": critic_dropout_key,
            },
            observation=timestep.obs,
            done=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            initial_carry=critic_carry,
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        return PPOState(
            step=0,
            timestep=timestep.from_sequence(),
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: PPOState, num_steps: int) -> PPOState:
        return state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: PPOState, num_steps: int) -> PPOState:
        num_outer_steps = num_steps // (self.cfg.num_envs * self.cfg.num_steps)
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(
        self, key: Key, state: PPOState, num_steps: int
    ) -> PPOState:
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        initial_actor_carry = self.actor_network.initialize_carry(
            (self.cfg.num_envs, None)
        )
        initial_critic_carry = self.critic_network.initialize_carry(
            (self.cfg.num_envs, None)
        )
        state = state.replace(
            timestep=timestep,
            actor_carry=initial_actor_carry,
            critic_carry=initial_critic_carry,
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )

        return state
