from typing import Any

import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.logger import Logger
from memory_rl.networks import Network, heads
from memory_rl.utils import compute_gae


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class PPOState:
    step: int
    obs: chex.Array
    env_state: chex.Array
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class PPO:
    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    actor: Network
    critic: Network
    optimizer: optax.GradientTransformation
    logger: Logger

    def init(self, key):
        key, env_key, actor_key, critic_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)
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

    def warmup(self, key, state, num_steps):
        """No warmup needed for PPO"""
        return key, state

    def _actor_loss_fn(self, params, transitions, advantages):
        probs = self.actor.apply(params, transitions.observation)
        log_prob = probs.log_prob(transitions.action)
        entropy = probs.entropy().mean()

        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss = -jnp.minimum(
            ratio * advantages,
            jnp.clip(
                ratio,
                1.0 - self.cfg.algorithm.clip_coef,
                1.0 + self.cfg.algorithm.clip_coef,
            )
            * advantages,
        ).mean()
        return actor_loss - self.cfg.algorithm.ent_coef * entropy, entropy

    def _critic_loss_fn(self, params, transitions, advantages, returns):
        value = self.critic.apply(
            params, transitions.observation
        ).squeeze(-1)

        if self.cfg.algorithm.clip_vloss:
            critic_loss = jnp.square(value - returns)
            clipped_value = transitions.value + jnp.clip(
                (value - transitions.value),
                -self.cfg.algorithm.clip_coef,
                self.cfg.algorithm.clip_coef,
            )
            clipped_critic_loss = jnp.square(clipped_value - returns)
            critic_loss = (
                0.5
                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
            )
        else:
            critic_loss = 0.5 * jnp.square(value - returns).mean()

        return self.cfg.algorithm.vf_coef * critic_loss

    def _update_minibatch(self, state, minibatch: tuple):
        transitions, advantages, returns = minibatch


        (actor_loss, entropy), actor_grads = jax.value_and_grad(self._actor_loss_fn, has_aux=True)(
            state.actor_params, transitions, advantages
        )
        actor_updates, actor_optimizer_state = self.optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(
            state.actor_params, actor_updates
        )

        critic_loss, critic_grads = jax.value_and_grad(self._critic_loss_fn)(
            state.critic_params, transitions, advantages, returns
        )
        critic_updates, critic_optimizer_state = self.optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(
            state.critic_params, critic_updates
        )

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return state, (actor_loss, critic_loss, entropy)

    def _update_epoch(self, carry: tuple, _):
        key, state, batch = carry


        key, permutation_key = jax.random.split(key)

        permutation = jax.random.permutation(
            permutation_key, self.cfg.algorithm.batch_size
        )
        flattened_batch = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), batch
        )
        shuffled_batch = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=0), flattened_batch
        )
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x, [self.cfg.algorithm.num_minibatches, -1] + list(x.shape[1:])
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

    def _step(self, carry: tuple, _):
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)

        probs = self.actor.apply(state.actor_params, state.obs)
        action = probs.sample(seed=action_key)
        log_prob = probs.log_prob(action)

        value = self.critic.apply(state.critic_params, state.obs).squeeze(-1)

        step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            observation=state.obs,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
            log_prob=log_prob,  # type: ignore
            value=value,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,
            env_state=env_state,
        )
        carry = (
            key,
            state,
        )
        return carry, transition

    def _update_step(self, carry: tuple, _):


        (key, state), transitions = jax.lax.scan(
            self._step,
            carry,
            length=self.cfg.algorithm.num_steps,
        )
        final_value = self.critic.apply(state.critic_params, state.obs).squeeze(-1)

        advantages, returns = compute_gae(
            self.cfg.algorithm.gamma,
            self.cfg.algorithm.gae_lambda,
            final_value,
            transitions,
        )

        if self.cfg.algorithm.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        batch = (transitions, advantages, returns)

        (key, state, batch), (actor_loss, critic_loss, entropy) = jax.lax.scan(
            self._update_epoch,
            (key, state, batch),
            length=self.cfg.algorithm.update_epochs,
        )
        transitions, *_ = batch

        def callback(logger, step, info, actor_loss, critic_loss, entropy):
            if info["returned_episode"].any():
                data = {
                    "training/episodic_returns": info[
                        "returned_episode_returns"
                    ].mean(),
                    "training/episodic_lengths": info[
                        "returned_episode_lengths"
                    ].mean(),
                    "losses/actor_loss": actor_loss.mean(),
                    "losses/critic_loss": critic_loss.mean(),
                    "losses/entropy": entropy.mean(),
                }
                logger.log(data, step=step)


        jax.debug.callback(
            callback, self.logger, state.step, transitions.info, actor_loss, critic_loss, entropy
        )

        return (key, state), transitions.info

    def train(self, key, state, num_steps):


        (key, state), info = jax.lax.scan(
            self._update_step,
            (key, state),
            length=num_steps
            // self.cfg.algorithm.num_envs
            // self.cfg.algorithm.num_steps,
        )

        return key, state, info

    def evaluate(self, key, state, num_steps):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        def step(carry, _):
            key, obs, env_state = carry

            probs = self.actor.apply(state.actor_params, obs)
            action = jnp.argmax(probs.logits, axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.algorithm.num_eval_envs)
            obs, env_state, _, _, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, env_state), info

        (key, obs, env_state), info = jax.lax.scan(
            step, (key, obs, env_state), length=num_steps
        )

        return key, info


def make_ppo(cfg, env, env_params, logger):

    actor = Network(
        feature_extractor=instantiate(cfg.algorithm.actor.feature_extractor),
        torso=instantiate(cfg.algorithm.actor.torso),
        head=heads.Categorical(
            action_dim=env.action_space(env_params).n,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        ),
    )

    critic = Network(
        feature_extractor=instantiate(cfg.algorithm.critic.feature_extractor),
        torso=instantiate(cfg.algorithm.critic.torso),
        head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
    )

    if cfg.algorithm.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=cfg.algorithm.learning_rate,
            end_value=0.0,
            transition_steps=(
                cfg.total_timesteps // cfg.algorithm.num_envs // cfg.algorithm.num_steps
            )
            * cfg.algorithm.update_epochs
            * cfg.algorithm.num_minibatches,
        )
    else:
        learning_rate = cfg.algorithm.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.algorithm.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    return PPO(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        logger=logger,
    )
