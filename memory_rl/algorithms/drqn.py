from functools import partial
from typing import Any

import chex
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.logger import Logger
from memory_rl.networks import RecurrentNetwork, heads
from memory_rl.utils import periodic_incremental_update


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
    q_network: RecurrentNetwork
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule
    logger: Logger

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        carry = self.q_network.initialize_carry(obs.shape)

        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=carry,
        )
        target_params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=carry,
        )
        optimizer_state = self.optimizer.init(params)

        transition = Transition(
            obs=obs[0],
            done=done[0],
            action=action[0],
            reward=reward[0],
            next_obs=obs[0],
            next_done=done[0],
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

            key, step_key, action_key, sample_key, memory_key = jax.random.split(key, 5)

            sample_key = jax.random.split(sample_key, self.cfg.algorithm.num_envs)
            random_action = jax.vmap(self.env.action_space(self.env_params).sample)(
                sample_key
            )

            next_hidden_state, q_values = self.q_network.apply(
                state.params,
                jnp.expand_dims(state.obs, 1),
                mask=jnp.expand_dims(state.done, 1),
                initial_carry=state.hidden_state,
                rngs={"memory": memory_key},
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
                action=jnp.expand_dims(action, 1),
                reward=jnp.expand_dims(reward, 1),
                next_obs=jnp.expand_dims(next_obs, 1),
                next_done=jnp.expand_dims(next_done, 1),
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

            # # Burn-in
            # burn_in_length = self.cfg.algorithm.mode.burn_in_length or 0
            # burn_in_length = min(burn_in_length, batch.experience.obs.shape[1])
            # initial_carry=jax.tree.map(
            #     lambda x: jnp.take(x, 0, axis=1), batch.experience.hidden_state
            # )
            # next_initial_carry=jax.tree.map(
            #     lambda x: jnp.take(x, 0, axis=1), batch.experience.next_hidden_state
            # )

            key, memory_key, next_memory_key = jax.random.split(key, 3)
            next_hidden_state, q_next_target = self.q_network.apply(
                state.target_params,
                batch.experience.next_obs,
                mask=batch.experience.next_done,
                return_carry_history=self.cfg.algorithm.update_hidden_state,
                rngs={"memory": next_memory_key},
            )
            q_next_target = jnp.max(q_next_target, axis=-1)

            next_q_value = (
                batch.experience.reward
                + (1 - batch.experience.next_done)
                * self.cfg.algorithm.gamma
                * q_next_target
            )

            mask = jnp.ones_like(next_q_value)
            if self.cfg.algorithm.mode.mask:
                episode_idx = jnp.cumsum(batch.experience.done, axis=1)
                terminal = (episode_idx == 1) & batch.experience.done
                mask *= (episode_idx == 0) | terminal

            # if self.cfg.algorithm.burn_in:
            #     pass

            def loss_fn(params):
                hidden_state, q_value = self.q_network.apply(
                    params,
                    batch.experience.obs,
                    mask=batch.experience.done,
                    return_carry_history=self.cfg.algorithm.update_hidden_state,
                    rngs={"memory": memory_key},
                )
                action = jnp.expand_dims(batch.experience.action, axis=-1)
                q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                td_error = q_value - next_q_value
                loss = jnp.square(td_error).mean(where=mask)
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

            def callback(logger, step, info, loss, q_value):
                if info["returned_episode"].any():
                    data = {
                        "training/episodic_returns": info["returned_episode_returns"][
                            info["returned_episode"]
                        ].mean(),
                        "training/episodic_lengths": info["returned_episode_lengths"][
                            info["returned_episode"]
                        ].mean(),
                        "losses/loss": loss,
                        "losses/q_value": q_value,
                    }
                    logger.log(data, step=step)

            jax.debug.callback(
                callback,
                self.logger,
                state.step,
                info,
                loss,
                q_value,
            )

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
        hidden_state = self.q_network.initialize_carry(obs.shape)

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
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    next_done: chex.Array


def make_drqn(cfg, env, env_params, logger) -> DRQN:

    q_network = RecurrentNetwork(
        feature_extractor=instantiate(cfg.algorithm.feature_extractor),
        torso=instantiate(cfg.algorithm.torso),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )

    sample_sequence_length = (
        cfg.algorithm.mode.length or env_params.max_steps_in_episode
    )
    buffer = instantiate(
        cfg.algorithm.buffer,
        sample_sequence_length=sample_sequence_length,
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
        logger=logger,
    )
