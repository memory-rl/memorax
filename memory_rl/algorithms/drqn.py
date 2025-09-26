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

from memory_rl.networks import RecurrentNetwork, heads
from memory_rl.utils import periodic_incremental_update, Transition


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
    start_e: float
    end_e: float
    exploration_fraction: float
    double: bool
    learning_starts: int
    train_frequency: int
    buffer: Any
    feature_extractor: Any
    torso: Any
    mode: Any


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

    def _greedy_action(
        self, key: chex.PRNGKey, state: DRQNState
    ) -> tuple[chex.PRNGKey, DRQNState, chex.Array]:
        key, memory_key = jax.random.split(key)
        hidden_state, q_values = self.q_network.apply(
            state.params,
            jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.hidden_state,
            rngs={"memory": memory_key},
        )
        action = jnp.argmax(q_values, axis=-1).squeeze(-1)
        state = state.replace(hidden_state=hidden_state)
        return key, state, action

    def _random_action(
        self, key: chex.PRNGKey, state: DRQNState
    ) -> tuple[chex.PRNGKey, DRQNState, chex.Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action

    def _epsilon_greedy_action(
        self, key: chex.PRNGKey, state: DRQNState
    ) -> tuple[chex.PRNGKey, DRQNState, chex.Array]:

        key, state, random_action = self._random_action(key, state)

        key, state, greedy_action = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, state, action

    def _step(
        self, carry, _, *, policy: Callable, write_to_buffer: bool = True
    ) -> tuple[chex.PRNGKey, DRQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action = policy(action_key, state)
        num_envs = state.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,  # type: ignore
            prev_done=state.done,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            next_obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            info=info,  # type: ignore
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            env_state=env_state,  # type: ignore
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(
        self, key: chex.PRNGKey, state: DRQNState
    ) -> tuple[DRQNState, chex.Array, chex.Array]:

        batch = self.buffer.sample(state.buffer_state, key)

        # # Burn-in
        # burn_in_length = self.cfg.mode.burn_in_length or 0
        # burn_in_length = min(burn_in_length, batch.experience.obs.shape[1])
        # initial_carry=jax.tree.map(
        #     lambda x: jnp.take(x, 0, axis=1), batch.experience.hidden_state
        # )
        # next_initial_carry=jax.tree.map(
        #     lambda x: jnp.take(x, 0, axis=1), batch.experience.next_hidden_state
        # )

        key, memory_key, next_memory_key = jax.random.split(key, 3)

        _, q_next_target = self.q_network.apply(
            state.target_params,
            batch.experience.next_obs,
            mask=batch.experience.done,
            rngs={"memory": next_memory_key},
        )

        if self.cfg.double:
            memory_key, double_memory_key = jax.random.split(memory_key)
            _, q_next_online = self.q_network.apply(
                state.params,
                batch.experience.next_obs,
                mask=batch.experience.done,
                rngs={"memory": double_memory_key},
            )
            next_actions = jnp.argmax(q_next_online, axis=-1)
            q_next_target = jnp.take_along_axis(
                q_next_target, jnp.expand_dims(next_actions, -1), axis=-1
            ).squeeze(-1)
        else:
            q_next_target = jnp.max(q_next_target, axis=-1)

        td_target = (
            batch.experience.reward
            + (1 - batch.experience.done) * self.cfg.gamma * q_next_target
        )

        mask = jnp.ones_like(td_target)
        if self.cfg.mode.mask:
            episode_idx = jnp.cumsum(batch.experience.prev_done, axis=1)
            terminal = (episode_idx == 1) & batch.experience.prev_done
            mask *= (episode_idx == 0) | terminal

        # if self.cfg.burn_in:
        #     pass
        def loss_fn(params):
            hidden_state, q_value = self.q_network.apply(
                params,
                batch.experience.obs,
                mask=batch.experience.prev_done,
                rngs={"memory": memory_key},
            )
            action = jnp.expand_dims(batch.experience.action, axis=-1)
            q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
            td_error = q_value - td_target
            loss = jnp.square(td_error).mean(where=mask.astype(jnp.bool_))
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
            self.cfg.target_network_frequency,
            self.cfg.tau,
        )

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
        )

        return state, loss, q_value.mean()

    def _learn(self, carry, _):

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, update_key = jax.random.split(key)
        state, loss, q_value = self._update(update_key, state)

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key):
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jax.vmap(self.env.action_space(self.env_params).sample)(env_keys)
        _, _, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        done = jnp.zeros_like(done)
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
            obs=obs,
            prev_done=done,
            action=action,
            reward=reward,
            next_obs=obs,
            done=done,
            info=info,
        )  # type: ignore
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

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

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def warmup(
        self, key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, DRQNState]:

        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def train(
        self,
        key: chex.PRNGKey,
        state: DRQNState,
        num_steps: int,
    ):

        (
            key,
            state,
        ), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )
        return key, state, transitions

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def evaluate(
        self, key: chex.PRNGKey, state: DRQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=jnp.bool_)
        hidden_state = self.q_network.initialize_carry(obs.shape)

        state = state.replace(obs=obs, done=done, hidden_state=hidden_state, env_state=env_state)  # type: ignore

        (key, _), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)


def make(cfg, env, env_params) -> DRQN:

    drqn_config = DRQNConfig(**cfg.algorithm)

    q_network = RecurrentNetwork(
        feature_extractor=instantiate(drqn_config.feature_extractor),
        torso=instantiate(drqn_config.torso),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )

    sample_sequence_length = min_length_time_axis = (
        drqn_config.mode.length or env_params.max_steps_in_episode
    )
    buffer = instantiate(
        drqn_config.buffer,
        sample_sequence_length=sample_sequence_length,
        min_length_time_axis=min_length_time_axis,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=drqn_config.learning_rate),
    )
    epsilon_schedule = optax.linear_schedule(
        drqn_config.start_e,
        drqn_config.end_e,
        int(drqn_config.exploration_fraction * cfg.total_timesteps),
        drqn_config.learning_starts,
    )

    return DRQN(
        cfg=drqn_config,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        buffer=buffer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
