from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.networks import Network, heads
from memory_rl.utils import periodic_incremental_update, Transition


@chex.dataclass
class Batch:
    """Data structure for a batch of transitions sampled from the replay buffer."""

    obs: chex.Array
    """Batch of obs with shape [batch_size, obs_dim]"""
    action: chex.Array
    """Batch of actions with shape [batch_size, action_dim]"""
    reward: chex.Array
    """Batch of rewards with shape [batch_size]"""
    done: chex.Array
    """Batch of done flags with shape [batch_size]"""


@chex.dataclass(frozen=True)
class SACDState:
    step: int
    env_state: gymnax.EnvState
    buffer_state: Any
    actor_params: core.FrozenDict[str, chex.ArrayTree]
    critic_params: core.FrozenDict[str, chex.ArrayTree]
    critic_target_params: core.FrozenDict[str, chex.ArrayTree]
    alpha_params: core.FrozenDict[str, chex.ArrayTree]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    obs: chex.Array


@chex.dataclass(frozen=True)
class SACD:
    cfg: DictConfig
    env: Any
    env_params: Any
    actor_network: nn.Module
    critic_network: nn.Module
    alpha_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    buffer: Any

    def _deterministic_action(self, key: chex.PRNGKey, state: SACDState) -> tuple[chex.PRNGKey, SACDState, chex.Array, chex.Array]:
        dist = self.actor_network.apply(state.actor_params, state.obs)
        action = dist.mode()
        return key, action

    def _stochastic_action(self, key: chex.PRNGKey, state: SACDState) -> tuple[chex.PRNGKey, SACDState, chex.Array, chex.Array]:
        key, sample_key = jax.random.split(key)
        dist = self.actor_network.apply(state.actor_params, state.obs)
        action = dist.sample(seed=sample_key)
        return key, action

    def _random_action(self, key: chex.PRNGKey, state: SACDState) -> tuple[chex.PRNGKey, SACDState, chex.Array, chex.Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.algorithm.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action

    def _step(self, carry, _, *, policy: Callable) -> tuple[chex.PRNGKey, SACDState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action = policy(action_key, state)
        step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
        )

        buffer_state = self.buffer.add(state.buffer_state, transition)
        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update_alpha(self, state: SACDState, batch: Batch):
        action_dim = self.env.action_space(self.env_params).n
        target_entropy = self.cfg.algorithm.target_entropy_scale * jnp.log(action_dim)

        def alpha_loss_fn(alpha_params):
            alpha = self.alpha_network.apply(alpha_params)
            log_alpha = jnp.log(alpha + 1e-8)
            dist = self.actor_network.apply(state.actor_params, batch.first.obs)
            entropy = dist.entropy().mean()
            alpha_loss = -log_alpha * (entropy - target_entropy)
            return alpha_loss, {"alpha": alpha, "alpha_loss": alpha_loss}

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

    def _update_actor(self, key, state: SACDState, batch: Batch):
        alpha = self.alpha_network.apply(state.alpha_params)

        def actor_loss_fn(actor_params):
            dist = self.actor_network.apply(actor_params, batch.first.obs)
            log_probs = jax.nn.log_softmax(dist.logits)
            q1, q2 = self.critic_network.apply(state.critic_params, batch.first.obs)
            q = jnp.minimum(q1, q2)
            actor_loss = (
                (dist.probs * ((alpha * log_probs) - q)).sum(axis=-1).mean()
            )
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": dist.entropy().mean(),
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

    def _update_critic(self, key, state: SACDState, batch: Batch):
        alpha = self.alpha_network.apply(state.alpha_params)
        dist = self.actor_network.apply(state.actor_params, batch.second.obs)
        log_probs = jax.nn.log_softmax(dist.logits)

        next_q1, next_q2 = self.critic_network.apply(
            state.critic_target_params, batch.second.obs
        )
        next_q = (
            dist.probs * (jnp.minimum(next_q1, next_q2) - alpha * log_probs)
        ).sum(axis=1)

        target_q = (
            batch.first.reward
            + self.cfg.algorithm.gamma * (1 - batch.first.done) * next_q
        )

        # if self.cfg.algorithm.backup_entropy:
        #     target_q -= (
        #         self.cfg.algorithm.gamma
        #         * (1 - batch.first.done)
        #         * alpha
        #         * next_log_probs
        #     )

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            q1, q2 = self.critic_network.apply(critic_params, batch.first.obs)
            q1_a = q1[jnp.arange(self.cfg.algorithm.batch_size), batch.first.action]
            q2_a = q2[jnp.arange(self.cfg.algorithm.batch_size), batch.first.action]
            critic_loss = ((q1_a - target_q) ** 2 + (q2_a - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
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
            self.cfg.algorithm.target_update_frequency,
            self.cfg.algorithm.tau,
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
            length=self.cfg.algorithm.train_frequency
            // self.cfg.algorithm.num_envs,
        )

        info = transitions.info
        key, update_key = jax.random.split(key)
        state, update_info = self._update(update_key, state)
        # info.update(update_info)

        return (key, state), info


    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key):
        key, env_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        # Initialize environment
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.algorithm.num_envs,),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        # Initialize actor
        actor_params = self.actor_network.init(actor_key, obs)
        actor_optimizer_state = self.actor_optimizer.init(actor_params)

        # Initialize critic
        critic_params = self.critic_network.init(critic_key, obs)
        critic_target_params = self.critic_network.init(critic_key, obs)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        # Initialize alpha
        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        transition = Transition(obs=obs, done=done, action=action, reward=reward, info=info)  # type: ignore
        transition = jax.tree.map(lambda x: x[0], transition)
        buffer_state = self.buffer.init(transition)

        return key, SACDState(
            step=0,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            obs=obs,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"])
    def warmup(
        self, key, state: SACDState, num_steps: int
    ) -> tuple[chex.PRNGKey, SACDState]:

        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action), (key, state), length=num_steps // self.cfg.algorithm.num_envs
        )
        return key, state


    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"])
    def train(self, key: chex.PRNGKey, state: SACDState, num_steps: int):
        (key, state), info = jax.lax.scan(
            self._update_step,
            (key, state),
            length=(num_steps // self.cfg.algorithm.train_frequency),
        )

        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"])
    def evaluate(self, key: chex.PRNGKey, state: SACDState, num_steps: int):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        state = state.replace(obs=obs, env_state=env_state)

        (key, _), transitions = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action), (key, state), length=num_steps
        )

        return key, transitions


def make_sacd(cfg, env, env_params) -> SACD:

    action_dim = env.action_space(env_params).n

    # Define networks
    actor_network = Network(
        feature_extractor=instantiate(cfg.algorithm.actor.feature_extractor),
        torso=instantiate(cfg.algorithm.actor.torso),
        head=heads.Categorical(
            action_dim=action_dim,
        ),
    )

    critic_network = nn.vmap(
        Network,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=2,
    )(
        feature_extractor=instantiate(cfg.algorithm.critic.feature_extractor),
        torso=instantiate(cfg.algorithm.critic.torso),
        head=heads.DiscreteQNetwork(env.action_space(env_params).n),
    )

    alpha_network = heads.Alpha(initial_alpha=cfg.algorithm.initial_alpha)

    # Define optimizers
    actor_optimizer = optax.adam(learning_rate=cfg.algorithm.actor_lr)
    critic_optimizer = optax.adam(learning_rate=cfg.algorithm.critic_lr)
    alpha_optimizer = optax.adam(learning_rate=cfg.algorithm.alpha_lr)

    buffer = fbx.make_flat_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=cfg.algorithm.batch_size,
        min_length=cfg.algorithm.batch_size,
        max_length=cfg.algorithm.buffer_size,
        add_sequences=False,
    )

    return SACD(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=alpha_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        buffer=buffer,
    )
