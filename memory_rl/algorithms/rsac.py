from functools import partial
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core
from flax.training import train_state
from hydra.utils import instantiate

from memory_rl.buffers import make_trajectory_buffer
from memory_rl.logger import Logger
from memory_rl.networks import RecurrentNetwork, heads
from memory_rl.utils import periodic_incremental_update


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


# Keep the network definitions (StochasticActor, Critic, DoubleCritic, Temperature) the same


@chex.dataclass(frozen=True)
class RSACState:
    step: int
    env_state: Any
    buffer_state: Any
    actor_params: core.FrozenDict[str, chex.ArrayTree]
    critic_params: core.FrozenDict[str, chex.ArrayTree]
    critic_target_params: core.FrozenDict[str, chex.ArrayTree]
    temp_params: core.FrozenDict[str, chex.ArrayTree]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    temp_optimizer_state: optax.OptState
    obs: chex.Array
    done: chex.Array
    hidden_state: chex.Array


@chex.dataclass(frozen=True)
class RSAC:
    cfg: Any  # Full config object
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
        done = jnp.zeros((self.cfg.algorithm.num_envs,))
        action = jnp.zeros(
            (
                self.cfg.algorithm.num_envs,
                self.env.action_space(self.env_params).shape[0],
            )
        )
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        # Initialize actor
        actor_params = self.actor_network.init(
            actor_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        actor_hidden_state = self.actor_network.initialize_carry(
            obs.shape,
        )

        # Initialize critic
        critic_params = self.critic_network.init(
            critic_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            jnp.expand_dims(action, 1),
        )
        critic_target_params = self.critic_network.init(
            critic_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
            jnp.expand_dims(action, 1),
        )
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        # Initialize temperature
        temp_params = self.temp_network.init(temp_key)
        temp_optimizer_state = self.temp_optimizer.init(temp_params)

        # Initialize buffer (placeholder - actual buffer initialization would go here)
        transition = Transition(obs=obs[0], done=done[0], action=action[0], reward=reward[0], next_obs=obs[0], next_done=done[0])  # type: ignore
        buffer_state = self.buffer.init(transition)

        return key, RSACState(
            step=0,
            hidden_state=actor_hidden_state,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            temp_params=temp_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            temp_optimizer_state=temp_optimizer_state,
            obs=obs,
            done=done,
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key, state: RSACState, num_steps: int
    ) -> tuple[chex.PRNGKey, RSACState]:
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
                obs=next_obs,  # type: ignore
                done=next_done,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (key, state), info

        (key, state), _ = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.algorithm.num_envs
        )
        return key, state

    @partial(jax.jit, static_argnames=["self"])
    def update_temperature(self, state: RSACState, entropy: float):
        action_dim = self.env.action_space(self.env_params).shape[0]
        target_entropy = -self.cfg.algorithm.target_entropy_multiplier * action_dim

        def temperature_loss_fn(temp_params):
            temperature = self.temp_network.apply(temp_params)
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
    def update_actor(self, key, state: RSACState, batch: Batch):
        temperature = self.temp_network.apply(state.temp_params)

        mask = jnp.ones_like(batch.reward)
        if self.cfg.algorithm.mode.mask:
            episode_idx = jnp.cumsum(batch.done, axis=1)
            terminal = (episode_idx == 1) & batch.done
            mask = (episode_idx == 0) | terminal

        def actor_loss_fn(actor_params):
            _, dist = self.actor_network.apply(actor_params, batch.obs, batch.done)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, (q1, q2) = self.critic_network.apply(
                state.critic_params, batch.obs, batch.done, actions
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * temperature - q.squeeze(-1)).mean(where=mask)

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
    def update_critic(self, key, state: RSACState, batch: Batch):

        # jax.debug.print('batch', batch)
        _, dist = self.actor_network.apply(
            state.actor_params, batch.next_obs, batch.next_done
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, (next_q1, next_q2) = self.critic_network.apply(
            state.critic_target_params, batch.next_obs, batch.next_done, next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)

        temperature = self.temp_network.apply(state.temp_params)
        target_q = batch.reward + self.cfg.algorithm.gamma * (
            1 - batch.next_done
        ) * next_q.squeeze(-1)

        if self.cfg.algorithm.backup_entropy:
            target_q -= (
                self.cfg.algorithm.gamma
                * (1 - batch.next_done)
                * temperature
                * next_log_probs
            )

        target_q = jax.lax.stop_gradient(target_q)

        mask = jnp.ones_like(target_q)
        if self.cfg.algorithm.mode.mask:
            episode_idx = jnp.cumsum(batch.done, axis=1)
            terminal = (episode_idx == 1) & batch.done
            mask = (episode_idx == 0) | terminal

        def critic_loss_fn(critic_params):
            _, (q1, q2) = self.critic_network.apply(
                critic_params, batch.obs, batch.done, batch.action
            )
            critic_loss = (
                (q1.squeeze(-1) - target_q) ** 2 + (q2.squeeze(-1) - target_q) ** 2
            ).mean(where=mask)

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
    def train(self, key: chex.PRNGKey, state: RSACState, num_steps: int):
        def step(carry, _):
            key, state = carry
            key, sample_key, action_key = jax.random.split(key, 3)

            # Sample action
            next_hidden_state, dist = self.actor_network.apply(
                state.actor_params,
                jnp.expand_dims(state.obs, 1),
                jnp.expand_dims(state.done, 1),
                initial_carry=state.hidden_state,
            )
            action = dist.sample(seed=sample_key).squeeze(1)

            # Step environment
            action_key = jax.random.split(action_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(action_key, state.env_state, action, self.env_params)

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
                hidden_state=next_hidden_state,
                env_state=env_state,
                buffer_state=buffer_state,
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
            state, temp_info = self.update_temperature(state, actor_info["entropy"])

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

            jax.debug.callback(callback, self.logger, state.step, info)

            return (key, state), info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=(num_steps // self.cfg.algorithm.train_frequency),
        )

        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: chex.PRNGKey, state: RSACState, num_steps: int):
        # Reinitialize environment state with num_eval_envs
        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_eval_envs)

        # Reset environments for evaluation
        eval_obs, eval_env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        eval_done = jnp.zeros((self.cfg.algorithm.num_eval_envs,), dtype=bool)

        # Initialize evaluation hidden state with correct size
        eval_hidden_state = self.actor_network.torso.initialize_carry(
            jax.random.key(0), eval_obs.shape
        )

        # Create evaluation state with reinitialized environment components
        eval_state = state.replace(
            obs=eval_obs,
            done=eval_done,
            env_state=eval_env_state,
            hidden_state=eval_hidden_state,
        )

        def step(carry, _):
            key, eval_state = carry
            key, sample_key, env_key = jax.random.split(key, 3)

            # Get action in evaluation mode (deterministic)
            next_hidden_state, dist = self.actor_network.apply(
                eval_state.actor_params,
                jnp.expand_dims(eval_state.obs, 1),
                jnp.expand_dims(eval_state.done, 1),
                initial_carry=eval_state.hidden_state,
                temperature=0.0,
            )
            action = dist.sample(seed=sample_key).squeeze(1)

            # Step environment with correct number of eval environments
            env_key = jax.random.split(env_key, self.cfg.algorithm.num_eval_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(env_key, eval_state.env_state, action, self.env_params)

            # Update evaluation state
            eval_state = eval_state.replace(
                obs=next_obs,
                done=next_done,
                hidden_state=next_hidden_state,
                env_state=env_state,
            )

            return (key, eval_state), info

        (key, eval_state), info = jax.lax.scan(
            step,
            (key, eval_state),
            length=num_steps // self.cfg.algorithm.num_eval_envs,
        )

        return key, info


def make_rsac(cfg, env, env_params, logger) -> RSAC:

    action_dim = env.action_space(env_params).shape[0]

    # Define networks
    actor_network = RecurrentNetwork(
        feature_extractor=instantiate(cfg.algorithm.actor.feature_extractor),
        torso=instantiate(cfg.algorithm.actor.torso),
        head=heads.SquashedGaussian(action_dim=action_dim),
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
        torso=instantiate(cfg.algorithm.critic.torso),
        head=heads.VNetwork(),
    )
    temp_network = heads.Temperature(initial_temperature=cfg.algorithm.init_temperature)

    # Define optimizers
    actor_optimizer = optax.adam(learning_rate=cfg.algorithm.policy_lr)
    critic_optimizer = optax.adam(learning_rate=cfg.algorithm.q_lr)
    temp_optimizer = optax.adam(learning_rate=cfg.algorithm.temp_lr)

    sample_sequence_length = min_length_time_axis = (
        cfg.algorithm.mode.length or env_params.max_steps_in_episode
    )
    buffer = instantiate(
        cfg.algorithm.mode.buffer,
        sample_sequence_length=sample_sequence_length,
        min_length_time_axis=min_length_time_axis,
    )

    return RSAC(
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
