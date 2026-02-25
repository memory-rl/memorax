from functools import partial
from typing import Any, Callable, Optional

import distrax
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
from memorax.utils import Timestep, Transition, memory_metrics
from memorax.utils.typing import Array, Discrete, Environment, EnvParams, EnvState, Key


@struct.dataclass(frozen=True)
class PPOCConfig:
    num_envs: int
    num_steps: int
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    num_options: int
    deliberation_cost: float = 0.01
    termination_coefficient: float = 0.5
    option_entropy_coefficient: float = 0.01
    target_kl: Optional[float] = None
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class PPOCState:
    step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Array
    current_option: Array


@struct.dataclass(frozen=True)
class PPOC:
    cfg: PPOCConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    def _deterministic_action(
        self, key: Key, state: PPOCState
    ) -> tuple[Key, PPOCState, Array, Array, None, dict, dict]:
        timestep = state.timestep.to_sequence()
        (actor_carry, ((intra_probs, termination_logits, option_logits), _)), intermediates = (
            self.actor_network.apply(
                state.actor_params,
                observation=timestep.obs,
                mask=timestep.done,
                action=timestep.action,
                reward=add_feature_axis(timestep.reward),
                done=timestep.done,
                initial_carry=state.actor_carry,
                mutable=["intermediates"],
            )
        )

        intra_probs = jax.tree.map(remove_time_axis, intra_probs)
        termination_logits = remove_time_axis(termination_logits)
        option_logits = remove_time_axis(option_logits)

        batch_idx = jnp.arange(termination_logits.shape[0])

        termination_probs = jax.nn.sigmoid(termination_logits)
        current_termination_prob = termination_probs[batch_idx, state.current_option]
        terminated = current_termination_prob > 0.5

        needs_new_option = terminated | state.timestep.done
        new_option = jnp.argmax(option_logits, axis=-1)
        current_option = jnp.where(needs_new_option, new_option, state.current_option)

        option_log_prob = distrax.Categorical(logits=option_logits).log_prob(
            current_option
        )

        intra_probs = jax.tree.map(lambda x: x[batch_idx, current_option], intra_probs)
        action = intra_probs.mode()
        log_prob = intra_probs.log_prob(action)

        state = state.replace(
            actor_carry=actor_carry,
            current_option=current_option,
        )

        option_metadata = {
            "option": current_option,
            "option_log_prob": option_log_prob,
            "termination_prob": current_termination_prob,
        }
        return key, state, action, log_prob, None, intermediates, option_metadata

    def _stochastic_action(
        self, key: Key, state: PPOCState
    ) -> tuple[Key, PPOCState, Array, Array, Array, dict, dict]:
        (
            key,
            action_key,
            option_key,
            termination_key,
            actor_memory_key,
            critic_memory_key,
        ) = jax.random.split(key, 6)

        timestep = state.timestep.to_sequence()
        (actor_carry, ((intra_probs, termination_logits, option_logits), _)), intermediates = (
            self.actor_network.apply(
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
        )

        intra_probs = jax.tree.map(remove_time_axis, intra_probs)
        termination_logits = remove_time_axis(termination_logits)
        option_logits = remove_time_axis(option_logits)

        batch_idx = jnp.arange(termination_logits.shape[0])

        # Check termination of current option
        termination_probs = jax.nn.sigmoid(termination_logits)
        current_termination_prob = termination_probs[batch_idx, state.current_option]
        terminated = jax.random.bernoulli(termination_key, current_termination_prob)

        # Select option
        needs_new_option = terminated | state.timestep.done
        option_probs = distrax.Categorical(logits=option_logits)
        sampled_option = option_probs.sample(seed=option_key)
        current_option = jnp.where(needs_new_option, sampled_option, state.current_option)
        option_log_prob = option_probs.log_prob(current_option)

        # Select action from intra-option policy
        intra_probs = jax.tree.map(lambda x: x[batch_idx, current_option], intra_probs)
        action, action_log_prob = intra_probs.sample_and_log_prob(seed=action_key)

        # Option-conditioned critic: Q(s, o) for all options
        critic_carry, (q_values, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.critic_carry,
            rngs={"memory": critic_memory_key},
        )
        # q_values: (B, 1, O, 1) -> (B, O)
        q_values = remove_time_axis(q_values)
        q_values = remove_feature_axis(q_values)
        value = q_values[batch_idx, current_option]

        state = state.replace(
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            current_option=current_option,
        )

        option_metadata = {
            "option": current_option,
            "option_log_prob": option_log_prob,
            "termination_prob": current_termination_prob,
            "q_values": q_values,
            "option_logits": option_logits,
        }
        return key, state, action, action_log_prob, value, intermediates, option_metadata

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
        key, state, action, log_prob, value, intermediates, option_metadata = policy(
            action_key, state
        )

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
        prev_action = jnp.where(
            jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        first = Timestep(
            obs=state.timestep.obs,
            action=prev_action,
            reward=jnp.where(state.timestep.done, 0, state.timestep.reward),
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
            metadata={**info, "intermediates": intermediates, **option_metadata},
            log_prob=log_prob,
            value=value,
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
        return (key, state), transition

    def _update_actor(
        self, key, state: PPOCState, initial_actor_carry, transitions, advantages, option_advantages
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
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
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            advantages = advantages[:, self.cfg.burn_in_length :]
            option_advantages = option_advantages[:, self.cfg.burn_in_length :]

        def actor_loss_fn(params):
            _, ((intra_probs, termination_logits, option_logits), _) = (
                self.actor_network.apply(
                    params,
                    observation=transitions.first.obs,
                    mask=transitions.first.done,
                    action=transitions.first.action,
                    reward=add_feature_axis(transitions.first.reward),
                    done=transitions.first.done,
                    initial_carry=initial_actor_carry,
                    rngs={"memory": memory_key, "dropout": dropout_key},
                )
            )

            options = transitions.metadata["option"]

            batch_size, time_steps = options.shape
            batch_idx = jnp.arange(batch_size)[:, None]
            time_idx = jnp.arange(time_steps)[None, :]

            # 1. Intra-option policy loss
            intra_probs = jax.tree.map(
                lambda x: x[batch_idx, time_idx, options], intra_probs
            )
            action_log_probs = intra_probs.log_prob(transitions.second.action)
            action_entropy = intra_probs.entropy().mean()

            action_ratio = jnp.exp(action_log_probs - transitions.log_prob)
            approx_kl = jnp.mean(transitions.log_prob - action_log_probs)
            clipfrac = jnp.mean(
                (jnp.abs(action_ratio - 1.0) > self.cfg.clip_coefficient).astype(
                    jnp.float32
                )
            )

            intra_loss = -jnp.minimum(
                action_ratio * advantages,
                jnp.clip(
                    action_ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()

            # 2. Option policy loss
            option_probs = distrax.Categorical(logits=option_logits)
            option_log_probs = option_probs.log_prob(options)
            option_entropy = option_probs.entropy().mean()

            old_option_log_probs = transitions.metadata["option_log_prob"]
            option_ratio = jnp.exp(option_log_probs - old_option_log_probs)
            option_loss = -jnp.minimum(
                option_ratio * advantages,
                jnp.clip(
                    option_ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()

            # 3. Termination loss (uses option advantages, not GAE advantages)
            termination_probs = jax.nn.sigmoid(termination_logits)
            selected_termination_probs = termination_probs[batch_idx, time_idx, options]
            termination_loss = (
                selected_termination_probs
                * (jax.lax.stop_gradient(option_advantages) + self.cfg.deliberation_cost)
            ).mean()

            total_loss = (
                intra_loss
                - self.cfg.entropy_coefficient * action_entropy
                + option_loss
                - self.cfg.option_entropy_coefficient * option_entropy
                + self.cfg.termination_coefficient * termination_loss
            )

            return total_loss, (
                action_entropy,
                approx_kl,
                clipfrac,
                option_entropy,
                option_loss,
                termination_loss,
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
        self, key, state: PPOCState, initial_critic_carry, transitions, returns
    ):
        key, memory_key, dropout_key = jax.random.split(key, 3)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], transitions
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
                lambda x: x[:, self.cfg.burn_in_length :], transitions
            )
            returns = returns[:, self.cfg.burn_in_length :]

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
            # values: (B, T, O, 1) -> (B, T, O) -> index by option -> (B, T)
            values = remove_feature_axis(values)
            options = transitions.metadata["option"]
            batch_size, time_steps = options.shape
            batch_idx = jnp.arange(batch_size)[:, None]
            time_idx = jnp.arange(time_steps)[None, :]
            values = values[batch_idx, time_idx, options]

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
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
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
            option_advantages,
        ) = minibatch

        key, state, critic_loss = self._update_critic(
            key, state, initial_critic_carry, transitions, returns
        )
        key, state, actor_loss, aux = self._update_actor(
            key, state, initial_actor_carry, transitions, advantages, option_advantages
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
            option_advantages,
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
            option_advantages,
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
                        (transitions, advantages, returns, option_advantages),
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

        (key, state), (
            actor_loss,
            critic_loss,
            (action_entropy, approx_kl, clipfrac, option_entropy, option_loss, termination_loss),
        ) = jax.lax.scan(
            self._update_minibatch,
            (key, state),
            minibatches,
        )

        metrics = jax.tree.map(
            lambda x: x.mean(),
            (
                actor_loss,
                critic_loss,
                action_entropy,
                approx_kl,
                clipfrac,
                option_entropy,
                option_loss,
                termination_loss,
            ),
        )

        return (
            key,
            state,
            initial_actor_carry,
            initial_critic_carry,
            transitions,
            advantages,
            returns,
            option_advantages,
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

        # Bootstrap: Q(s, o) for all options, select current option for GAE
        timestep = state.timestep.to_sequence()
        _, (bootstrap_q, _) = self.critic_network.apply(
            state.critic_params,
            observation=timestep.obs,
            mask=timestep.done,
            action=timestep.action,
            reward=add_feature_axis(timestep.reward),
            done=timestep.done,
            initial_carry=state.critic_carry,
        )
        # bootstrap_q: (B, 1, O, 1) -> (B, O)
        bootstrap_q = remove_time_axis(bootstrap_q)
        bootstrap_q = remove_feature_axis(bootstrap_q)
        value = bootstrap_q[jnp.arange(bootstrap_q.shape[0]), state.current_option]

        _, advantages = jax.lax.scan(
            self._generalized_advantage_estimation,
            (jnp.zeros_like(value), value),
            transitions,
            reverse=True,
            unroll=16,
        )
        returns = advantages + transitions.value

        # Compute option advantages: A_Ω(s,o) = Q(s,o) - V_Ω(s)
        q_values = transitions.metadata["q_values"]          # (T, B, O)
        option_logits = transitions.metadata["option_logits"]  # (T, B, O)
        option_probs = jax.nn.softmax(option_logits, axis=-1)
        v_omega = jnp.sum(option_probs * q_values, axis=-1)   # (T, B)
        options = transitions.metadata["option"]               # (T, B)
        time_idx = jnp.arange(q_values.shape[0])[:, None]
        batch_idx = jnp.arange(q_values.shape[1])[None, :]
        q_selected = q_values[time_idx, batch_idx, options]    # (T, B)
        option_advantages = q_selected - v_omega               # (T, B)

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        advantages = jnp.swapaxes(advantages, 0, 1)
        returns = jnp.swapaxes(returns, 0, 1)
        option_advantages = jnp.swapaxes(option_advantages, 0, 1)

        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def cond_fun(carry):
            *_, (_, _, _, approx_kl, _, _, _, _), epoch = carry

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
                option_advantages,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                0,
            ),
        )

        (
            actor_loss,
            critic_loss,
            action_entropy,
            approx_kl,
            clipfrac,
            option_entropy,
            option_loss,
            termination_loss,
        ) = jax.tree.map(lambda x: jnp.expand_dims(x, axis=(0, 1)), metrics)

        memory = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=(0, 1)),
            memory_metrics(state.actor_carry, initial_actor_carry),
        )
        metadata = {
            **transitions.metadata,
            "losses/actor_loss": actor_loss,
            "losses/critic_loss": critic_loss,
            "losses/entropy": action_entropy,
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": clipfrac,
            "losses/option_entropy": option_entropy,
            "losses/option_loss": option_loss,
            "losses/termination_loss": termination_loss,
            **memory,
        }

        return (
            key,
            state,
        ), transitions.replace(
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

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        return (
            key,
            PPOCState(
                step=0,
                timestep=timestep.from_sequence(),
                actor_carry=actor_carry,
                critic_carry=critic_carry,
                env_state=env_state,
                actor_params=actor_params,
                critic_params=critic_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
                current_option=jnp.zeros((self.cfg.num_envs,), dtype=jnp.int32),
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: PPOCState, num_steps: int) -> tuple[Key, PPOCState]:
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: Key, state: PPOCState, num_steps: int):
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
    def evaluate(self, key: Key, state: PPOCState, num_steps: int, deterministic=True):
        key, reset_key = jax.random.split(key)
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
            current_option=jnp.zeros((self.cfg.num_envs,), dtype=jnp.int32),
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
