import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from hydra.utils import instantiate
from networks import RecurrentDoubleCritic, RecurrentStochasticActor, Temperature
from omegaconf import OmegaConf
from recurrent_networks import MaskedGRUCell
from utils.base_types import OnlineAndTargetState, RNNOffPolicyLearnerState

import wandb
from utils import (
    BraxGymnaxWrapper,
    LogWrapper,
    make_trajectory_buffer,
    periodic_incremental_update,
)


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
class RSACConfig:
    """Configuration for SAC"""

    name: str
    policy_lr: float
    q_lr: float
    temp_lr: float
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    gamma: float
    tau: float
    train_frequency: int
    target_update_frequency: int
    batch_size: int
    hidden_dims: tuple[int]
    actor_cell_size: int
    critic_cell_size: int
    sample_sequence_length: int
    init_temperature: float
    policy_log_std_min: float
    policy_log_std_max: float
    policy_final_fc_init_scale: float
    target_entropy_multiplier: float
    max_action: float
    backup_entropy: bool
    learning_starts: int
    track: bool


@chex.dataclass(frozen=True)
class RSACState(RNNOffPolicyLearnerState):
    actor: train_state.TrainState
    critic: OnlineAndTargetState
    temp: train_state.TrainState
    obs: chex.Array
    done: chex.Array


@chex.dataclass(frozen=True)
class RSAC:
    cfg: RSACConfig
    env: Any
    env_params: Any
    actor_network: nn.Module
    critic_network: nn.Module
    temp_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    temp_optimizer: optax.GradientTransformation
    buffer: Any

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, actor_key, critic_key, temp_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        # Initialize environment
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros((self.cfg.num_envs,))
        action = jnp.zeros(
            (self.cfg.num_envs, self.env.action_space(self.env_params).shape[0])
        )
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        # Initialize actor
        actor_params = self.actor_network.init(
            actor_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(done, 1),
        )["params"]
        actor = train_state.TrainState.create(
            apply_fn=self.actor_network.apply,
            params=actor_params,
            tx=self.actor_optimizer,
        )
        actor_hidden_state = self.actor_network.cell.initialize_carry(
            jax.random.key(0), (self.cfg.num_envs, self.cfg.hidden_dims[-1])
        )

        # Initialize critic
        critic_params = self.critic_network.init(
            critic_key,
            jnp.expand_dims(obs, 1),
            jnp.expand_dims(action, 1),
            jnp.expand_dims(done, 1),
        )["params"]
        critic = OnlineAndTargetState.create(
            apply_fn=self.critic_network.apply,
            params=critic_params,
            target_params=critic_params,  # Initialize target with same params
            tx=self.critic_optimizer,
            opt_state=self.critic_optimizer.init(critic_params),
        )

        # Initialize temperature
        temp_params = self.temp_network.init(temp_key)["params"]
        temp = train_state.TrainState.create(
            apply_fn=self.temp_network.apply, params=temp_params, tx=self.temp_optimizer
        )

        # Initialize buffer (placeholder - actual buffer initialization would go here)
        transition = Transition(obs=obs[0], done=done[0], action=action[0], reward=reward[0], next_obs=obs[0], next_done=done[0])  # type: ignore
        buffer_state = self.buffer.init(transition)

        return key, RSACState(
            step=0,
            key=key,
            hidden_state=actor_hidden_state,
            env_state=env_state,
            buffer_state=buffer_state,
            actor=actor,
            critic=critic,
            temp=temp,
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

            sample_key = jax.random.split(sample_key, self.cfg.num_envs)
            action = jax.vmap(self.env.action_space(self.env_params).sample)(sample_key)

            step_key = jax.random.split(step_key, self.cfg.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=jnp.expand_dims(state.obs, 1),  # type: ignore
                done=jnp.expand_dims(done, 1),  # type: ignore
                action=jnp.expand_dims(action, 1),  # type: ignore
                reward=jnp.expand_dims(reward, 1),  # type: ignore
                next_obs=jnp.expand_dims(next_obs, 1),  # type: ignore
                next_done=jnp.expand_dims(done, 1),  # type: ignore
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                obs=next_obs,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (key, state), info

        (key, state), _ = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.num_envs
        )
        return key, state

    @partial(jax.jit, static_argnames=["self"])
    def update_temperature(self, state: RSACState, entropy: float):
        action_dim = self.env.action_space(self.env_params).shape[0]
        target_entropy = -self.cfg.target_entropy_multiplier * action_dim

        def temperature_loss_fn(temp_params):
            temperature = state.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temp_loss": temp_loss}

        (_, info), grads = jax.value_and_grad(temperature_loss_fn, has_aux=True)(
            state.temp.params
        )
        new_temp = state.temp.apply_gradients(grads=grads)

        return new_temp, info

    @partial(jax.jit, static_argnames=["self"])
    def update_actor(self, key, state: RSACState, batch: Batch):
        temperature = state.temp.apply_fn({"params": state.temp.params})

        def actor_loss_fn(actor_params):
            _, dist = state.actor.apply_fn(
                {"params": actor_params}, batch.obs, batch.done
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, q1, q2 = state.critic.apply_fn(
                {"params": state.critic.params}, batch.obs, actions, batch.done
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * temperature - q).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        (_, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor.params
        )
        new_actor = state.actor.apply_gradients(grads=grads)

        return new_actor, info

    @partial(jax.jit, static_argnames=["self"])
    def update_critic(self, key, state: RSACState, batch: Batch):

        # jax.debug.print('batch', batch)
        _, dist = state.actor.apply_fn(
            {"params": state.actor.params}, batch.next_obs, batch.next_done
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, next_q1, next_q2 = state.critic.apply_fn(
            {"params": state.critic.target_params},
            batch.next_obs,
            next_actions,
            batch.next_done,
        )
        next_q = jnp.minimum(next_q1, next_q2)

        temperature = state.temp.apply_fn({"params": state.temp.params})
        target_q = batch.reward + self.cfg.gamma * (1 - batch.done) * next_q

        if self.cfg.backup_entropy:
            target_q -= self.cfg.gamma * (1 - batch.done) * temperature * next_log_probs

        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(critic_params):
            _, q1, q2 = state.critic.apply_fn(
                {"params": critic_params}, batch.obs, batch.action, batch.done
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        (_, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic.params
        )
        new_critic = state.critic.apply_gradients(grads=grads)

        # Update target network periodically
        new_target_params = periodic_incremental_update(
            new_critic.params,
            state.critic.target_params,
            state.step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        new_critic = new_critic.replace(target_params=new_target_params)
        return new_critic, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(self, key: chex.PRNGKey, state: RSACState, num_steps: int):
        def step(carry, _):
            key, state = carry
            key, sample_key, action_key = jax.random.split(key, 3)

            # Sample action
            next_hidden_state, dist = state.actor.apply_fn(
                {"params": state.actor.params},
                jnp.expand_dims(state.obs, 1),
                jnp.expand_dims(state.done, 1),
                initial_carry=state.hidden_state,
            )
            action = dist.sample(seed=action_key).squeeze(1)

            # Step environment
            action_key = jax.random.split(action_key, self.cfg.num_envs)
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
                step=state.step + self.cfg.num_envs,
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
            new_critic, critic_info = self.update_critic(
                critic_key, state, batch.experience
            )  ### Because fbx has weird way of storing transitions

            # Update actor using updated critic
            key, actor_key = jax.random.split(key)
            temp_state = state.replace(critic=new_critic)
            new_actor, actor_info = self.update_actor(
                actor_key, temp_state, batch.experience
            )  ### Because fbx has weird way of storing transitions

            # Update temperature using updated actor and critic
            temp_state = temp_state.replace(actor=new_actor)
            new_temp, temp_info = self.update_temperature(
                temp_state, actor_info["entropy"]
            )

            # Update state
            state = state.replace(actor=new_actor, critic=new_critic, temp=new_temp)

            # info = {**critic_info, **actor_info, **temp_info}
            info = {}
            return state, info

        def update_step(carry, _):
            (key, state), info = jax.lax.scan(
                step, carry, length=self.cfg.train_frequency // self.cfg.num_envs
            )

            key, update_key = jax.random.split(key)

            state, update_info = update(update_key, state)

            if self.cfg.track:

                def callback(step, info):
                    if step % 100 == 0:
                        wandb.log(
                            {
                                "training/episodic_return": info[
                                    "returned_episode_returns"
                                ].mean(),
                                "training/episodic_length": info[
                                    "returned_episode_lengths"
                                ].mean(),
                            },
                            step=step,
                        )

                jax.debug.callback(callback, state.step, info)

            info.update(update_info)
            return (key, state), info

        (key, state), info = jax.lax.scan(
            update_step, (key, state), length=(num_steps // self.cfg.train_frequency)
        )

        return key, state, info

    @partial(jax.jit, static_argnames=["self"])
    def sample(
        self,
        state: RSACState,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        hidden_state: Any = None,
    ):
        key, sample_key = jax.random.split(state.key)
        hidden_state, dist = state.actor.apply_fn(
            {"params": state.actor.params}, obs, done, initial_carry=hidden_state
        )
        action = dist.sample(seed=sample_key)
        new_state = state.replace(key=key)
        return hidden_state, new_state, action

    @partial(jax.jit, static_argnames=["self"])
    def sample_eval(
        self,
        key: chex.PRNGKey,
        state: RSACState,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        hidden_state: Any = None,
    ):
        hidden_state, dist = state.actor.apply_fn(
            {"params": state.actor.params},
            obs,
            done,
            initial_carry=hidden_state,
            temperature=0.0,
        )
        action = dist.sample(seed=key)
        return hidden_state, action

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: chex.PRNGKey, state: RSACState, num_steps: int):
        def step(carry, _):
            key, state = carry
            key, sample_key, env_key = jax.random.split(key, 3)

            # Get action in evaluation mode (deterministic)
            next_hidden_state, action = self.sample_eval(
                sample_key,
                state,
                jnp.expand_dims(state.obs, 1),
                jnp.expand_dims(state.done, 1),
                state.hidden_state,
            )

            action = action.squeeze(1)

            # Step environment
            env_key = jax.random.split(env_key, self.cfg.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(env_key, state.env_state, action, self.env_params)

            # Update state
            state = state.replace(
                obs=next_obs,
                done=next_done,
                hidden_state=next_hidden_state,
                env_state=env_state,
            )

            return (key, state), info

        (key, state), info = jax.lax.scan(
            step, (key, state), length=num_steps // self.cfg.num_envs
        )

        return key, info


def make_rsac(cfg) -> RSAC:

    env = BraxGymnaxWrapper(cfg.environment.env_id, backend="mjx")
    env_params = None
    # env, env_params = gymnax.make(cfg.environment.env_id)
    env = LogWrapper(env)
    action_dim = env.action_space(env_params).shape[0]

    # Define networks
    actor_network = RecurrentStochasticActor(
        cell=MaskedGRUCell(cfg.algorithm.actor_cell_size),
        hidden_dims=cfg.algorithm.hidden_dims,
        action_dim=action_dim,
        max_action=cfg.algorithm.max_action,
        final_fc_init_scale=cfg.algorithm.policy_final_fc_init_scale,
        log_std_min=cfg.algorithm.policy_log_std_min,
        log_std_max=cfg.algorithm.policy_log_std_max,
    )

    critic_network = RecurrentDoubleCritic(
        cell=MaskedGRUCell(cfg.algorithm.critic_cell_size),
        hidden_dims=cfg.algorithm.hidden_dims,
    )

    temp_network = Temperature(initial_temperature=cfg.algorithm.init_temperature)

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

    algorithm_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    algorithm_cfg["hidden_dims"] = tuple(algorithm_cfg["hidden_dims"])
    return RSAC(
        cfg=RSACConfig(**algorithm_cfg, track=cfg.logger.track),  # type: ignore
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        temp_network=temp_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temp_optimizer=temp_optimizer,
        buffer=buffer,
    )
