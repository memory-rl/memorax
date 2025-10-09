from typing import Any

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
)
from memory_rl.networks import RecurrentNetwork
from memory_rl.utils import generalized_advantage_estimatation, Transition


@chex.dataclass(frozen=True)
class RPPOContinuousConfig:
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
class RPPOContinuousState:
    step: int
    obs: Array
    done: Array
    env_state: EnvState
    actor_hidden_state: Array
    critic_hidden_state: Array
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class RPPOContinuous:
    cfg: RPPOContinuousConfig
    env: Environment
    env_params: EnvParams
    actor_network: RecurrentNetwork
    critic_network: RecurrentNetwork
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    def init(self, key):
        key, env_key, actor_key, critic_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros(self.cfg.num_envs, dtype=bool)

        # Initialize hidden states using the torso from the networks
        actor_hidden_state = self.actor_network.initialize_carry(
            (self.cfg.num_envs, self.cfg.actor.torso.features),
        )
        critic_hidden_state = self.critic_network.initialize_carry(
            (self.cfg.num_envs, self.cfg.critic.torso.features),
        )

        dummy_obs_for_init = jnp.expand_dims(obs, 1)  # (num_envs, 1, obs_dim)
        dummy_mask_for_init = jnp.expand_dims(done, 1)  # (num_envs, 1)

        actor_params = self.actor_network.init(
            actor_key,
            dummy_obs_for_init,
            mask=dummy_mask_for_init,
            initial_carry=actor_hidden_state,
        )

        critic_params = self.critic_network.init(
            critic_key,
            dummy_obs_for_init,
            mask=dummy_mask_for_init,
            initial_carry=critic_hidden_state,
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        return (
            key,
            RPPOContinuousState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                actor_hidden_state=actor_hidden_state,  # type: ignore
                critic_hidden_state=critic_hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
                actor_params=actor_params,  # type: ignore
                critic_params=critic_params,  # type: ignore
                actor_optimizer_state=actor_optimizer_state,  # type: ignore
                critic_optimizer_state=critic_optimizer_state,  # type: ignore
            ),
        )

    def warmup(self, key, state, num_steps):
        """No warmup needed for RPPO"""
        return key, state

    def train(self, key, state, num_steps):

        def update_step(carry: tuple, _):

            def step(carry: tuple, _):
                key, state = carry

                key, action_key, step_key = jax.random.split(key, 3)

                # Actor forward pass
                actor_h_next, dist = self.actor_network.apply(
                    state.actor_params,
                    jnp.expand_dims(state.obs, 1),  # (B, 1, F_obs)
                    mask=jnp.expand_dims(state.done, 1),  # (B, 1) mask
                    initial_carry=state.actor_hidden_state,
                )

                # Sample action and get log prob
                action, log_prob = dist.sample_and_log_prob(seed=action_key)
                action = action.squeeze(1)  # Remove time dimension
                log_prob = log_prob.squeeze(1)  # Remove time dimension

                # Critic forward pass - needs actions for Q-value estimation
                critic_h_next, value = self.critic_network.apply(
                    state.critic_params,
                    jnp.expand_dims(state.obs, 1),  # (B, 1, F_obs)
                    mask=jnp.expand_dims(state.done, 1),  # (B, 1) mask
                    initial_carry=state.critic_hidden_state,
                )

                value = value.squeeze((1, -1))

                step_key = jax.random.split(step_key, self.cfg.num_envs)
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
                    done=done,
                    actor_hidden_state=actor_h_next,
                    critic_hidden_state=critic_h_next,
                )
                return (key, state), transition

            key, state = carry
            initial_actor_h_rollout = state.actor_hidden_state
            initial_critic_h_rollout = state.critic_hidden_state
            (key, state), transitions = jax.lax.scan(
                step,
                (key, state),
                length=self.cfg.num_steps,
            )

            # For final value, we need the final action to compute Q-value
            # Sample action from current policy for final state
            key, final_action_key = jax.random.split(key)
            _, final_dist = self.actor_network.apply(
                state.actor_params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.actor_hidden_state,
            )
            final_action = final_dist.sample(seed=final_action_key).squeeze(1)

            _, final_value = self.critic_network.apply(
                state.critic_params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.critic_hidden_state,
            )

            final_value = final_value.squeeze((1, -1))

            # Compute GAE on time-major data (T, B, ...)
            advantages, returns = generalized_advantage_estimatation(
                self.cfg.gamma,
                self.cfg.gae_lambda,
                transitions,
                final_value,
                state.done,
            )

            # Convert from time_major=True format (T, B, ...) to time_major=False format (B, T, ...)
            transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

            # Convert advantages and returns from (T, B) to (B, T) format
            advantages = jnp.swapaxes(advantages, 0, 1)
            returns = jnp.swapaxes(returns, 0, 1)

            if self.cfg.norm_adv:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5  # Increased epsilon for numerical stability
                )

            def update_epoch(carry: tuple, _):

                def update_minibatch(state, minibatch: tuple):
                    (
                        initial_actor_h_mb,
                        initial_critic_h_mb,
                        transitions,
                        advantages,
                        returns,
                    ) = minibatch

                    def loss_fn(
                        params,
                        initial_actor_h,
                        initial_critic_h,
                        transitions,
                        advantages,
                        returns,
                    ):
                        actor_params = params["actor"]
                        critic_params = params["critic"]

                        # Actor forward pass
                        _, dist = self.actor_network.apply(
                            actor_params,
                            transitions.observation,  # (B,T,F)
                            mask=transitions.done,  # (B,T) - mask
                            initial_carry=initial_actor_h,  # (B,H)
                        )

                        # Critic forward pass
                        _, value = self.critic_network.apply(
                            critic_params,
                            transitions.observation,  # (B,T,F)
                            mask=transitions.done,  # (B,T) - mask
                            initial_carry=initial_critic_h,  # (B,H)
                        )
                        value = value.squeeze(-1)

                        log_prob = dist.log_prob(transitions.action)
                        entropy = -log_prob.mean()

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

                        if self.cfg.clip_vloss:
                            critic_loss = jnp.square(value - returns)
                            clipped_value = transitions.value + jnp.clip(
                                (value - transitions.value),
                                -self.cfg.clip_coef,
                                self.cfg.clip_coef,
                            )
                            clipped_critic_loss = jnp.square(clipped_value - returns)
                            critic_loss = (
                                0.5
                                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
                            )
                        else:
                            critic_loss = 0.5 * jnp.square(value - returns).mean()

                        # # Add numerical stability checks
                        # critic_loss = jnp.nan_to_num(critic_loss, nan=0.0, posinf=10.0, neginf=0.0)
                        # actor_loss = jnp.nan_to_num(actor_loss, nan=0.0, posinf=10.0, neginf=0.0)
                        # entropy = jnp.nan_to_num(entropy, nan=0.0, posinf=1.0, neginf=-1.0)

                        loss = (
                            actor_loss
                            + self.cfg.vf_coef * critic_loss
                            - self.cfg.ent_coef * entropy
                        )

                        return loss, (actor_loss, critic_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss, (actor_loss, critic_loss, entropy_loss)), grads = grad_fn(
                        {"actor": state.actor_params, "critic": state.critic_params},
                        initial_actor_h_mb,  # (B,H)
                        initial_critic_h_mb,  # (B,H)
                        transitions,
                        advantages,
                        returns,
                    )

                    actor_grads = grads["actor"]
                    critic_grads = grads["critic"]

                    actor_updates, new_actor_opt_state = self.actor_optimizer.update(
                        actor_grads, state.actor_optimizer_state, state.actor_params
                    )
                    new_actor_params = optax.apply_updates(
                        state.actor_params, actor_updates
                    )

                    critic_updates, new_critic_opt_state = self.critic_optimizer.update(
                        critic_grads, state.critic_optimizer_state, state.critic_params
                    )
                    new_critic_params = optax.apply_updates(
                        state.critic_params, critic_updates
                    )

                    state = state.replace(
                        actor_params=new_actor_params,
                        critic_params=new_critic_params,
                        actor_optimizer_state=new_actor_opt_state,
                        critic_optimizer_state=new_critic_opt_state,
                    )
                    return state, (loss, actor_loss, critic_loss, entropy_loss)

                (
                    key,
                    state,
                    initial_actor_h_epoch,
                    initial_critic_h_epoch,
                    transitions,
                    advantages,
                    returns,
                ) = carry

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(permutation_key, self.cfg.num_envs)

                batch = (
                    initial_actor_h_epoch,
                    initial_critic_h_epoch,
                    transitions,
                    advantages,
                    returns,
                )

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x,
                        [self.cfg.num_minibatches, -1] + list(x.shape[1:]),
                    ),
                    shuffled_batch,
                )

                state, loss = jax.lax.scan(
                    update_minibatch,
                    state,
                    minibatches,
                )

                return (
                    key,
                    state,
                    initial_actor_h_epoch,
                    initial_critic_h_epoch,
                    transitions,
                    advantages,
                    returns,
                ), loss

            (key, state, *_), losses = jax.lax.scan(
                update_epoch,
                (
                    key,
                    state,
                    initial_actor_h_rollout,  # (B,H)
                    initial_critic_h_rollout,  # (B,H)
                    transitions,
                    advantages,
                    returns,
                ),
                length=self.cfg.update_epochs,
            )

            return (
                key,
                state,
            ), transitions.info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=num_steps // (self.cfg.num_envs * self.cfg.num_steps),
        )

        return key, state, info

    def evaluate(self, key, state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=bool)
        initial_actor_hidden_state = self.actor_network.initialize_carry(
            (self.cfg.num_eval_envs, self.cfg.actor.torso.features),
        )

        def step(carry, _):
            key, obs, done, env_state, actor_h_state = carry

            next_actor_h_state, dist = self.actor_network.apply(
                state.actor_params,
                jnp.expand_dims(obs, 1),  # (num_eval_envs, 1, obs_dim)
                mask=jnp.expand_dims(done, 1),  # (num_eval_envs, 1)
                initial_carry=actor_h_state,
            )
            action = dist.mode().squeeze(1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.num_eval_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            transition = Transition(
                obs=obs,  # type: ignore
                action=action,  # type: ignore
                reward=reward,  # type: ignore
                done=done,  # type: ignore
                next_obs=next_obs,  # type: ignore
                info=info,  # type: ignore
            )

            return (key, next_obs, done, env_state, next_actor_h_state), transition

        (key, *_), transitions = jax.lax.scan(
            step,
            (key, obs, done, env_state, initial_actor_hidden_state),
            length=num_steps // self.cfg.num_eval_envs,
        )

        return key, transitions
