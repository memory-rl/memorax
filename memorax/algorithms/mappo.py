from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition
from memorax.utils.axes import (
    add_feature_axis,
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.typing import Array, Environment, EnvParams, EnvState, Key, Carry, PyTree

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
    target_kl: float | None = None
    burn_in_length: int = 0

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

    def _deterministic_action(
        self, key: Key, state: MAPPOState
    ) -> tuple[MAPPOState, Array, Array, None, dict]:
        timestep = to_sequence(state.timestep)

        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_carry,
            mutable=["intermediates"],
        )

        action = jnp.argmax(probs.logits, axis=-1)
        log_prob = probs.log_prob(action)

        action = jax.vmap(remove_time_axis)(action)
        log_prob = jax.vmap(remove_time_axis)(log_prob)

        state = state.replace(actor_carry=actor_carry)
        return state, action, log_prob, None, intermediates

    def _stochastic_action(
        self, key: Key, state: MAPPOState
    ) -> tuple[MAPPOState, Array, Array, Array, dict]:
        action_key, actor_torso_key, critic_torso_key = jax.random.split(key, 3)
        timestep = to_sequence(state.timestep)

        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.actor_carry,
            rngs={"torso": actor_torso_key},
            mutable=["intermediates"],
        )

        action_keys = jax.random.split(action_key, self.env.num_agents)
        sampled_action, log_prob = jax.vmap(lambda p, k: p.sample_and_log_prob(seed=k))(
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
            rngs={"torso": critic_torso_key},
        )

        action = jax.vmap(remove_time_axis)(sampled_action)
        log_prob = jax.vmap(remove_time_axis)(log_prob)
        value = jax.vmap(remove_time_axis)(value)
        value = remove_feature_axis(value)

        state = state.replace(actor_carry=actor_carry, critic_carry=critic_carry)
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
        self, state: MAPPOState, key: Key, *, policy: Callable
    ) -> tuple[MAPPOState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value, intermediates = policy(action_key, state)

        _, num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 1), out_axes=(1, 0, 1, 1, 0)
        )(step_keys, state.env_state, action)
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
            obs=next_obs,
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
            step=state.step + num_envs,
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
        self, key: Key, state: MAPPOState, initial_actor_carry: Carry, transitions: Transition, advantages: Array
    ) -> tuple[MAPPOState, Array, tuple[Array, Array, Array]]:
        torso_key, dropout_key = jax.random.split(key)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )

            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                burn_in.first.obs,
                burn_in.first.done,
                burn_in.first.action,
                add_feature_axis(burn_in.first.reward),
                burn_in.first.done,
                initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, :, self.cfg.burn_in_length :]

        def actor_loss_fn(params: PyTree):
            _, (probs, _) = self.actor_network.apply(
                params,
                transitions.first.obs,
                transitions.first.done,
                transitions.first.action,
                add_feature_axis(transitions.first.reward),
                transitions.first.done,
                initial_actor_carry,
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
        self, key: Key, state: MAPPOState, initial_critic_carry: Carry, transitions: Transition, returns: Array
    ) -> tuple[MAPPOState, Array]:
        torso_key, dropout_key = jax.random.split(key)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, :, : self.cfg.burn_in_length], transitions
            )

            initial_critic_carry, (_, _) = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                burn_in.first.obs,
                burn_in.first.done,
                burn_in.first.action,
                add_feature_axis(burn_in.first.reward),
                burn_in.first.done,
                initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)
            transitions = jax.tree.map(
                lambda x: x[:, :, self.cfg.burn_in_length :], transitions
            )
            returns = returns[:, :, self.cfg.burn_in_length :]

        def critic_loss_fn(params: PyTree):
            _, (values, aux) = self.critic_network.apply(
                params,
                transitions.first.obs,
                transitions.first.done,
                transitions.first.action,
                add_feature_axis(transitions.first.reward),
                transitions.first.done,
                initial_critic_carry,
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
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return state, critic_loss.mean()

    def _update_minibatch(
        self, state: MAPPOState, xs: tuple
    ) -> tuple[MAPPOState, tuple[Array, Array, tuple[Array, Array, Array]]]:
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

    def _update_step(self, state: MAPPOState, key: Key) -> tuple[MAPPOState, None]:
        step_key, epoch_key = jax.random.split(key)

        initial_actor_carry = state.actor_carry
        initial_critic_carry = state.critic_carry

        step_keys = jax.random.split(step_key, self.cfg.num_steps)
        state, transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            state,
            step_keys,
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

        _, advantages = jax.lax.scan(
            self._generalized_advantage_estimation,
            (jnp.zeros_like(value), value),
            transitions,
            reverse=True,
            unroll=16,
        )
        returns = advantages + transitions.aux["value"]

        transitions = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, min(2, x.ndim - 1)), transitions
        )
        advantages = jnp.moveaxis(advantages, 0, 2)
        returns = jnp.moveaxis(returns, 0, 2)

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
    def init(self, key: Key) -> MAPPOState:
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
        ) = jax.random.split(key, 7)

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
                "torso": actor_torso_key,
                "dropout": actor_dropout_key,
            },
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            actor_carry,
        )

        critic_carry = self.critic_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "torso": critic_torso_key,
                "dropout": critic_dropout_key,
            },
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            critic_carry,
        )

        return MAPPOState(
            step=0,
            timestep=from_sequence(timestep),
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_optimizer_state=self.actor_optimizer.init(actor_params),
            critic_optimizer_state=self.critic_optimizer.init(critic_params),
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: Key, state: MAPPOState, num_steps: int
    ) -> MAPPOState:
        return state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: MAPPOState, num_steps: int) -> MAPPOState:
        num_outer_steps = num_steps // (self.cfg.num_envs * self.cfg.num_steps)
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: MAPPOState, num_steps: int) -> MAPPOState:
        reset_key, eval_key = jax.random.split(key)
        num_agents = self.env.num_agents

        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, out_axes=(1, 0))(reset_keys)

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
        critic_carry = self.critic_network.initialize_carry(
            (num_agents, self.cfg.num_envs, None)
        )

        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            env_state=env_state,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )

        return state
