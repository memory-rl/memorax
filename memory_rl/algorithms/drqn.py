import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import core
from gymnax.wrappers.purerl import FlattenObservationWrapper
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from memory_rl.networks import RNN, Network, heads
from memory_rl.utils import (LogWrapper, make_trajectory_buffer,
                             periodic_incremental_update)


# TODO: REMOVE CONFIGS
@chex.dataclass(frozen=True)
class DRQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    gamma: float
    tau: float
    target_network_frequency: int
    batch_size: int
    cell: dict = field(hash=False)
    sample_sequence_length: int
    update_hidden_state: bool
    start_e: float
    end_e: float
    exploration_fraction: float
    train_frequency: int
    learning_starts: int
    feature_extractor: dict = field(hash=False)
    track: bool


@chex.dataclass(frozen=True)
class DRQNState:
    step: int
    obs: chex.Array
    done: chex.Array
    hidden_state: tuple
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: Any


@chex.dataclass(frozen=True)
class DRQN:
    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        carry = self.q_network.torso.initialize_carry(
            (self.cfg.algorithm.num_envs, self.cfg.algorithm.cell.features)
        )

        params = self.q_network.init(
            q_key,
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=carry,
        )
        target_params = self.q_network.init(
            q_key,
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=carry,
        )
        optimizer_state = self.optimizer.init(params)

        transition = Transition(
            obs=obs[0],
            done=done[0],
            hidden_state=(
                (carry[0][0], carry[1][0]) if isinstance(carry, tuple) else carry[0]
            ),
            action=action[0],
            reward=reward[0],
            next_obs=obs[0],
            next_done=done[0],
            next_hidden_state=(
                (carry[0][0], carry[1][0]) if isinstance(carry, tuple) else carry[0]
            ),
        )  # type: ignore
        buffer_state = self.buffer.init(transition)

        return (
            key,
            DRQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                hidden_state=carry,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                target_params=target_params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                buffer_state=buffer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, DRQNState]:

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
                hidden_state=jax.tree.map(  # type: ignore
                    lambda x: jnp.expand_dims(x, 1), state.hidden_state
                ),
                action=jnp.expand_dims(action, 1),  # type: ignore
                reward=jnp.expand_dims(reward, 1),  # type: ignore
                next_obs=jnp.expand_dims(next_obs, 1),  # type: ignore
                next_done=jnp.expand_dims(next_done, 1),  # type: ignore
                next_hidden_state=jax.tree.map(  # type: ignore
                    lambda x: jnp.expand_dims(x, 1), state.hidden_state
                ),
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

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: chex.PRNGKey,
        state: DRQNState,
        num_steps: int,
    ):

        def step(carry: tuple, _):

            (
                key,
                state,
            ) = carry

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, self.cfg.algorithm.num_envs)
            random_action = jax.vmap(self.env.action_space(self.env_params).sample)(
                sample_key
            )

            next_hidden_state, q_values = self.q_network.apply(
                state.params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.hidden_state,
            )
            greedy_action = q_values.squeeze(1).argmax(axis=-1)

            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape)
                < self.epsilon_schedule(state.step),
                random_action,
                greedy_action,
            )

            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, next_done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            transition = Transition(
                obs=jnp.expand_dims(state.obs, 1),
                done=jnp.expand_dims(state.done, 1),
                hidden_state=jax.tree.map(
                    lambda x: jnp.expand_dims(x, 1),
                    state.hidden_state,
                ),
                action=jnp.expand_dims(action, 1),
                reward=jnp.expand_dims(reward, 1),
                next_obs=jnp.expand_dims(next_obs, 1),
                next_done=jnp.expand_dims(next_done, 1),
                next_hidden_state=jax.tree.map(
                    lambda x: jnp.expand_dims(x, 1),
                    next_hidden_state,
                ),
            )

            buffer_state = self.buffer.add(state.buffer_state, transition)
            state = state.replace(
                step=state.step + self.cfg.algorithm.num_envs,
                obs=next_obs,  # type: ignore
                done=next_done,  # type: ignore
                hidden_state=next_hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
                buffer_state=buffer_state,
            )

            return (
                key,
                state,
            ), info

        def update(
            key: chex.PRNGKey, state: DRQNState
        ) -> tuple[DRQNState, chex.Array, chex.Array]:

            batch = self.buffer.sample(state.buffer_state, key)

            next_hidden_state, q_next_target = self.q_network.apply(
                state.target_params,
                batch.experience.next_obs,
                mask=batch.experience.next_done,
                initial_carry=jax.tree.map(
                    lambda x: x[:, 0, :], batch.experience.next_hidden_state
                ),
                return_carry_history=self.cfg.algorithm.update_hidden_state,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.experience.reward
                + (1 - batch.experience.next_done)
                * self.cfg.algorithm.gamma
                * q_next_target
            )

            def loss_fn(params):
                hidden_state, q_value = self.q_network.apply(
                    params,
                    batch.experience.obs,
                    mask=batch.experience.done,
                    initial_carry=jax.tree.map(
                        lambda x: x[:, 0, :], batch.experience.hidden_state
                    ),
                    return_carry_history=self.cfg.algorithm.update_hidden_state,
                )
                action = jnp.expand_dims(batch.experience.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                loss = jnp.square(q_value - next_q_value).mean()
                return loss, (q_value, hidden_state)

            (loss, (q_value, hidden_state)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
            updates, optimizer_state = self.optimizer.update(
                grads, state.optimizer_state, state.params
            )
            params = optax.apply_updates(state.params, updates)
            target_params = periodic_incremental_update(
                params,
                state.target_params,
                state.step,
                self.cfg.algorithm.target_network_frequency,
                self.cfg.algorithm.tau,
            )

            if self.cfg.algorithm.update_hidden_state:
                hidden_state = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1), hidden_state
                )
                next_hidden_state = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1), next_hidden_state
                )
                experience = batch.experience.replace(
                    hidden_state=hidden_state,
                    next_hidden_state=next_hidden_state,
                )
                state = state.replace(
                    buffer_state=self.buffer.update(
                        state.buffer_state,
                        experience,
                        batch.sampled_batch_indices,
                        batch.traj_time_indices,
                    )
                )

            state = state.replace(
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
            )

            return state, loss, q_value.mean()

        def learn(carry, _):

            (key, state), info = jax.lax.scan(
                step,
                carry,
                length=self.cfg.algorithm.train_frequency
                // self.cfg.algorithm.num_envs,
            )

            key, update_key = jax.random.split(key)
            state, loss, q_value = update(update_key, state)

            if self.cfg.logger.track:

                def callback(step, info, loss, q_value):
                    if step % 100 == 0:
                        wandb.log(
                            {
                                "training/episodic_return": info[
                                    "returned_episode_returns"
                                ].mean(),
                                "training/episodic_length": info[
                                    "returned_episode_lengths"
                                ].mean(),
                                "losses/loss": loss,
                                "losses/q_value": q_value,
                            },
                            step=step,
                        )

                jax.debug.callback(callback, state.step, info, loss, q_value)

            return (key, state), info

        (
            key,
            state,
        ), info = jax.lax.scan(
            learn,
            (key, state),
            length=(num_steps // self.cfg.algorithm.train_frequency),
        )
        return key, state, info

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(
        self, key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_envs, dtype=jnp.bool)
        hidden_state = self.q_network.torso.initialize_carry(
            (self.cfg.algorithm.num_envs, self.cfg.algorithm.cell.features)
        )

        state = state.replace(obs=obs, done=done, hidden_state=hidden_state, env_state=env_state)  # type: ignore

        def step(
            carry: tuple[chex.PRNGKey, DRQNState],
            _,
        ):
            key, state = carry

            hidden_state, q_values = self.q_network.apply(
                state.params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.hidden_state,
            )
            action = q_values.squeeze(1).argmax(axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            obs, env_state, _, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, state.env_state, action, self.env_params)

            state = state.replace(
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                hidden_state=hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
            )

            return (key, state), info

        (key, _), info = jax.lax.scan(
            step,
            (key, state),
            length=num_steps,
        )

        return key, info


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    done: chex.Array
    hidden_state: tuple
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    next_done: chex.Array
    next_hidden_state: tuple


def make_drqn(cfg, env, env_params) -> DRQN:

    q_network = Network(
        feature_extractor=instantiate(cfg.algorithm.feature_extractor),
        torso=RNN(
            cell=instantiate(cfg.algorithm.cell),
        ),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )
    buffer = make_trajectory_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=cfg.algorithm.batch_size,
        sample_sequence_length=cfg.algorithm.sample_sequence_length,
        period=1,
        min_length_time_axis=cfg.algorithm.sample_sequence_length,
        max_size=cfg.algorithm.buffer_size,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=cfg.algorithm.learning_rate),
    )
    epsilon_schedule = optax.linear_schedule(
        cfg.algorithm.start_e,
        cfg.algorithm.end_e,
        int(cfg.algorithm.exploration_fraction * cfg.total_timesteps),
        cfg.algorithm.learning_starts,
    )

    return DRQN(
        cfg=cfg,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
