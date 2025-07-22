from functools import partial
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
from memory_rl.networks import RecurrentNetwork, heads
from memory_rl.utils import make_trajectory_buffer, periodic_incremental_update


@chex.dataclass
class Batch:
    """Data structure for a batch of transitions sampled from the replay buffer."""

    obs: chex.Array
    """Batch of obs with shape [batch_size, obs_dim]"""
    done: chex.Array
    """Batch of done flags with shape [batch_size]"""
    action: chex.Array
    """Batch of actions with shape [batch_size, action_dim]"""
    reward: chex.Array
    """Batch of rewards with shape [batch_size]"""
    next_obs: chex.Array
    """Batch of next obs with shape [batch_size, obs_dim]"""
    next_done: chex.Array
    """Batch of next done flags with shape [batch_size]"""


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    done: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    next_done: chex.Array


@chex.dataclass(frozen=True)
class RSACDState:
    step: int
    env_state: gymnax.EnvState
    buffer_state: Any
    actor_params: core.FrozenDict[str, chex.ArrayTree]
    critic_params: core.FrozenDict[str, chex.ArrayTree]
    critic_target_params: core.FrozenDict[str, chex.ArrayTree]
    temp_params: core.FrozenDict[str, chex.ArrayTree]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    actor_hidden_state: Any
    critic_hidden_state: Any
    temp_optimizer_state: optax.OptState
    obs: chex.Array
    done: chex.Array


@chex.dataclass(frozen=True)
class RSACD:
    cfg: DictConfig
    env: Any
    env_params: Any
    actor_network: nn.Module
    critic_network: nn.Module
    temp_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    temp_optimizer: optax.GradientTransformation
    buffer: Any
    logger: Logger

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, actor_key, critic_key, temp_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        # Initialize environment
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.algorithm.num_envs,),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        # Initialize actor
        actor_params = self.actor_network.init(
            actor_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        actor_hidden_state = self.actor_network.initialize_carry(
            (self.cfg.algorithm.num_envs, self.cfg.algorithm.actor.torso.cell.features)
        )

        # Initialize critic
        critic_params = self.critic_network.init(
            critic_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        critic_target_params = self.critic_network.init(
            critic_key, jnp.expand_dims(obs, 1), jnp.expand_dims(done, 1)
        )
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        critic_hidden_state = self.critic_network.initialize_carry(
            (self.cfg.algorithm.num_envs, self.cfg.algorithm.critic.torso.cell.features)
        )

        # Initialize temperature
        temp_params = self.temp_network.init(temp_key)
        temp_optimizer_state = self.temp_optimizer.init(temp_params)

        transition = Transition(obs=obs[0], done=done[0], action=action[0], reward=reward[0], next_obs=obs[0], next_done=done[0])  # type: ignore
        buffer_state = self.buffer.init(transition)

        return key, RSACDState(
            step=0,
            obs=obs,
            done=done,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            temp_params=temp_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            temp_optimizer_state=temp_optimizer_state,
            actor_hidden_state=actor_hidden_state,
            critic_hidden_state=critic_hidden_state,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key, state: RSACDState, num_steps: int
    ) -> tuple[chex.PRNGKey, RSACDState]:
        def step(carry, _):

            key, state = carry

            key, sample_key, step_key = jax.random.split(key, 3)

            sample_key = jax.random.split(sample_key, self.cfg.algorithm.num_envs)
            action = jax.vmap(self.env.action_space(self.env_params).sample)(sample_key)

            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=jnp.expand_dims(state.obs, 1),  # type: ignore
                done=jnp.expand_dims(state.done, 1),  # type: ignore
                action=jnp.expand_dims(action, 1),  # type: ignore
                reward=jnp.expand_dims(reward, 1),  # type: ignore
                next_obs=jnp.expand_dims(next_obs, 1),  # type: ignore
                next_done=jnp.expand_dims(next_done, 1),  # type: ignore
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                step=state.step + self.cfg.algorithm.num_envs,
                obs=next_obs,
                env_state=env_state,
                buffer_state=buffer_state,
            )

            return (key, state), info

        (key, state), _ = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.algorithm.num_envs
        )
        return key, state

    @partial(jax.jit, static_argnames=["self"])
    def update_temperature(self, state: RSACDState, batch: Batch):
        action_dim = self.env.action_space(self.env_params).n
        target_entropy = -self.cfg.algorithm.target_entropy_multiplier * action_dim

        def temperature_loss_fn(temp_params):
            temperature = self.temp_network.apply(temp_params)

            _, dist = self.actor_network.apply(
                state.actor_params, batch.obs, batch.done
            )
            entropy = dist.entropy().mean()
            temp_loss = temperature * (entropy - target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temp_loss": temp_loss}

        (_, info), grads = jax.value_and_grad(temperature_loss_fn, has_aux=True)(
            state.temp_params
        )
        updates, optimizer_state = self.temp_optimizer.update(
            grads, state.temp_optimizer_state, state.temp_params
        )
        temp_params = optax.apply_updates(state.temp_params, updates)

        state = state.replace(
            temp_params=temp_params, temp_optimizer_state=optimizer_state
        )

        return state, info

    @partial(jax.jit, static_argnames=["self"])
    def update_actor(self, key, state: RSACDState, batch: Batch):
        temperature = self.temp_network.apply(state.temp_params)

        def actor_loss_fn(actor_params):
            _, dist = self.actor_network.apply(actor_params, batch.obs, batch.done)
            log_probs = jax.nn.log_softmax(dist.logits)
            _, (q1, q2) = self.critic_network.apply(
                state.critic_params, batch.obs, batch.done
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (
                (dist.probs * (temperature * log_probs - q)).sum(axis=-1).mean()
            )
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

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

    @partial(jax.jit, static_argnames=["self"])
    def update_critic(self, key, state: RSACDState, batch: Batch):
        temperature = self.temp_network.apply(state.temp_params)
        _, dist = self.actor_network.apply(
            state.actor_params, batch.next_obs, batch.next_done
        )
        log_probs = jax.nn.log_softmax(dist.logits)

        _, (next_q1, next_q2) = self.critic_network.apply(
            state.critic_target_params, batch.next_obs, batch.next_done
        )
        next_q = (
            dist.probs * (jnp.minimum(next_q1, next_q2) - temperature * log_probs)
        ).sum(axis=-1)

        target_q = (
            batch.reward + self.cfg.algorithm.gamma * (1 - batch.next_done) * next_q
        )

        # if self.cfg.algorithm.backup_entropy:
        #     target_q -= (
        #         self.cfg.algorithm.gamma
        #         * (1 - batch.done)
        #         * temperature
        #         * next_log_probs
        #     )

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            _, (q1, q2) = self.critic_network.apply(
                critic_params, batch.obs, batch.done
            )
            idx = batch.action[..., None]  # [B, T, 1]
            q1_a = jnp.take_along_axis(q1, idx, axis=-1).squeeze(-1)
            q2_a = jnp.take_along_axis(q2, idx, axis=-1).squeeze(-1)
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

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: chex.PRNGKey, state: RSACDState, num_steps: int):
        def step(carry, _):
            key, state = carry
            key, sample_key, step_key = jax.random.split(key, 3)

            # Sample action
            actor_hidden_state, dist = self.actor_network.apply(
                state.actor_params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.actor_hidden_state,
            )
            action = dist.sample(seed=sample_key).squeeze(1)

            # Step environment
            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            # Add to buffer
            transition = Transition(
                obs=jnp.expand_dims(state.obs, 1),  # type: ignore
                done=jnp.expand_dims(state.done, 1),  # type: ignore
                action=jnp.expand_dims(action, 1),  # type: ignore
                reward=jnp.expand_dims(reward, 1),  # type: ignore
                next_obs=jnp.expand_dims(next_obs, 1),  # type: ignore
                next_done=jnp.expand_dims(next_done, 1),  # type: ignore
            )
            buffer_state = self.buffer.add(state.buffer_state, transition)

            # Update state
            state = state.replace(
                step=state.step + self.cfg.algorithm.num_envs,
                obs=next_obs,
                done=next_done,
                env_state=env_state,
                buffer_state=buffer_state,
                actor_hidden_state=actor_hidden_state,
            )

            return (key, state), info

        def update(key, state):
            # Sample from buffer
            key, batch_key = jax.random.split(key)
            batch = self.buffer.sample(state.buffer_state, batch_key)

            # Update critic
            key, critic_key = jax.random.split(key)
            state, critic_info = self.update_critic(
                critic_key, state, batch.experience
            )  ### Because fbx has weird way of storing transitions

            # Update actor using updated critic
            key, actor_key = jax.random.split(key)
            state, actor_info = self.update_actor(
                actor_key, state, batch.experience
            )  ### Because fbx has weird way of storing transitions

            # Update temperature using updated actor and critic
            state, temp_info = self.update_temperature(state, batch.experience)

            info = {**critic_info, **actor_info, **temp_info}
            return state, info

        def update_step(carry, _):
            (key, state), info = jax.lax.scan(
                step,
                carry,
                length=self.cfg.algorithm.train_frequency
                // self.cfg.algorithm.num_envs,
            )

            key, update_key = jax.random.split(key)
            state, update_info = update(update_key, state)
            info.update(update_info)

            def callback(logger, step, info):
                if info["returned_episode"].any():
                    data = {
                        "training/episodic_return": info[
                            "returned_episode_returns"
                        ].mean(),
                        "training/episodic_length": info[
                            "returned_episode_lengths"
                        ].mean(),
                    }
                    logger.log(data, step=step)

            return (key, state), info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=(num_steps // self.cfg.algorithm.train_frequency),
        )

        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: chex.PRNGKey, state: RSACDState, num_steps: int):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_eval_envs, dtype=jnp.bool)
        actor_hidden_state = self.actor_network.initialize_carry(
            (
                self.cfg.algorithm.num_eval_envs,
                self.cfg.algorithm.actor.torso.cell.features,
            )
        )
        state = state.replace(
            obs=obs,
            done=done,
            actor_hidden_state=actor_hidden_state,
            env_state=env_state,
        )

        def step(carry, _):
            key, state = carry

            actor_hidden_state, dist = self.actor_network.apply(
                state.actor_params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.actor_hidden_state,
            )
            action = dist.mode().squeeze(1)

            # Step environment
            key, env_key = jax.random.split(key)
            env_key = jax.random.split(env_key, self.cfg.algorithm.num_eval_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(env_key, state.env_state, action, self.env_params)

            # Update state
            state = state.replace(
                obs=next_obs,
                done=next_done,
                actor_hidden_state=actor_hidden_state,
                env_state=env_state,
            )

            return (key, state), info

        (key, state), info = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.algorithm.num_eval_envs
        )

        return key, info


def make_rsacd(cfg, env, env_params, logger) -> RSACD:

    action_dim = env.action_space(env_params).n

    # Define networks
    actor_network = RecurrentNetwork(
        feature_extractor=instantiate(cfg.algorithm.actor.feature_extractor),
        cell=instantiate(cfg.algorithm.actor.torso.cell),
        head=heads.Categorical(
            action_dim=action_dim,
            kernel_init=instantiate(cfg.algorithm.actor.head.kernel_init),
        ),
    )

    critic_network = nn.vmap(
        RecurrentNetwork,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=2,
    )(
        feature_extractor=instantiate(cfg.algorithm.critic.feature_extractor),
        cell=instantiate(cfg.algorithm.critic.torso.cell),
        head=heads.DiscreteQNetwork(env.action_space(env_params).n),
    )

    temp_network = heads.Temperature(initial_temperature=cfg.algorithm.init_temperature)

    # Define optimizers
    actor_optimizer = optax.adam(learning_rate=cfg.algorithm.policy_lr)
    critic_optimizer = optax.adam(learning_rate=cfg.algorithm.q_lr)
    temp_optimizer = optax.adam(learning_rate=cfg.algorithm.temp_lr)

    buffer = make_trajectory_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=cfg.algorithm.batch_size,
        sample_sequence_length=cfg.algorithm.sample_sequence_length,
        period=1,
        min_length_time_axis=cfg.algorithm.sample_sequence_length,
        max_size=cfg.algorithm.buffer_size,
    )

    return RSACD(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        temp_network=temp_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temp_optimizer=temp_optimizer,
        buffer=buffer,
        logger=logger,
    )
