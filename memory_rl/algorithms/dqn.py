from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.networks import Network, heads
from memory_rl.utils import Transition


@chex.dataclass(frozen=True)
class DQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    buffer_size: int
    gamma: float
    tau: float
    target_network_frequency: int
    batch_size: int
    start_e: float
    end_e: float
    exploration_fraction: float
    learning_starts: int
    train_frequency: int
    buffer: Any
    feature_extractor: Any
    torso: Any
    double: bool


@chex.dataclass(frozen=True)
class DQNState:
    """
    Immutable container for training state of DQN algorithm.
    """

    step: int
    obs: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: fbx.trajectory_buffer.TrajectoryBufferState


@chex.dataclass(frozen=True)
class DQN:
    """
    Deep Q-Network (DQN) reinforcement learning algorithm.
    """

    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    buffer: fbx.trajectory_buffer.TrajectoryBuffer
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: chex.PRNGKey, state: DQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action

    def _random_action(
        self, key: chex.PRNGKey, state: DQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action

    def _epsilon_greedy_action(
        self, key: chex.PRNGKey, state: DQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:

        key, random_action = self._random_action(key, state)

        key, greedy_action = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, action

    def _step(
        self, carry, _, *, policy: Callable, write_to_buffer: bool = True
    ) -> tuple[chex.PRNGKey, DQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action = policy(action_key, state)

        num_envs = state.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
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

        buffer_state = state.buffer_state

        if write_to_buffer:
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(
        self, key: chex.PRNGKey, state: DQNState
    ) -> tuple[DQNState, chex.Array, chex.Array]:

        key, sample_key = jax.random.split(key)
        batch = self.buffer.sample(state.buffer_state, sample_key).experience

        next_target_q_values = self.q_network.apply(
            state.target_params,
            batch.second.obs,
        )

        if self.cfg.double:
            next_q_values = self.q_network.apply(
                state.params,
                batch.second.obs,
            )
            next_action = jnp.argmax(next_q_values, axis=-1)
            next_target_q_value = jnp.take_along_axis(
                next_target_q_values, jnp.expand_dims(next_action, -1), axis=-1
            ).squeeze(-1)
        else:
            next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        td_target = (
            batch.first.reward
            + (1 - batch.first.done) * self.cfg.gamma * next_target_q_value
        )

        def loss_fn(params):
            q_value = self.q_network.apply(
                params,
                batch.first.obs,
            )
            action = jnp.expand_dims(batch.first.action, axis=-1)
            q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
            loss = jnp.square(q_value - td_target).mean()
            return loss, q_value

        (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        target_params = optax.periodic_update(
            optax.incremental_update(params, state.target_params, self.cfg.tau),
            state.target_params,
            state.step,
            self.cfg.target_network_frequency,
        )

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
        )

        return key, state, loss, q_value.mean()

    def _learn(
        self, carry: tuple[chex.PRNGKey, DQNState], _
    ) -> tuple[tuple[chex.PRNGKey, DQNState], dict]:

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, state, loss, q_value = self._update(key, state)

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[chex.PRNGKey, DQNState, chex.Array, gymnax.EnvState]:
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )

        params = self.q_network.init(q_key, obs)
        target_params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        transition = Transition(obs=obs, action=action, reward=reward, done=done, info=info)  # type: ignore
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return (
            key,
            DQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                target_params=target_params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                buffer_state=buffer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: chex.PRNGKey, state: DQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, DQNState]:

        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: chex.PRNGKey,
        state: DQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, DQNState, dict]:
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(
        self, key: chex.PRNGKey, state: DQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, Any]:
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        state = state.replace(obs=obs, env_state=env_state)

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)


def make(cfg, env, env_params) -> DQN:
    """
    Factory function to construct a DQN agent from Args.

    Args:
        args: Experiment configuration.

    Returns:
        An initialized DQN instance ready for training.
    """

    dqn_config = DQNConfig(**cfg.algorithm)

    q_network = Network(
        feature_extractor=instantiate(dqn_config.feature_extractor),
        torso=instantiate(dqn_config.torso),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )

    buffer = instantiate(dqn_config.buffer)
    optimizer = optax.adam(learning_rate=dqn_config.learning_rate)
    epsilon_schedule = optax.linear_schedule(
        dqn_config.start_e,
        dqn_config.end_e,
        int(dqn_config.exploration_fraction * cfg.total_timesteps),
        dqn_config.learning_starts,
    )
    return DQN(
        cfg=dqn_config,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
