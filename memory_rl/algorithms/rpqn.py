from functools import partial
from typing import Any, Callable

import gymnax
import jax
import flax.linen as nn
import jax.numpy as jnp
import optax
from flax import core, struct

from memory_rl.networks import RecurrentNetwork
from memory_rl.utils import Transition
from memory_rl.utils.typing import Array, Environment, EnvParams, EnvState, Key


@struct.dataclass(frozen=True)
class RPQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    num_steps: int
    gamma: float
    td_lambda: float
    num_minibatches: int
    update_epochs: int
    td_lambda: float
    max_grad_norm: float
    learning_starts: int
    start_e: float
    end_e: float
    exploration_fraction: float
    feature_extractor: nn.Module
    torso: nn.Module

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class RPQNState:
    """
    Immutable container for training state of RPQN algorithm.
    """

    step: int
    obs: Array
    done: Array
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    hidden_state: Array
    optimizer_state: optax.OptState


@struct.dataclass(frozen=True)
class RPQN:
    """
    Deep Q-Network (RPQN) reinforcement learning algorithm.
    """

    cfg: RPQNConfig
    env: Environment
    env_params: EnvParams
    q_network: RecurrentNetwork
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: RPQNState
    ) -> tuple[Key, RPQNState, Array]:
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
        return key, state, action, q_values

    def _random_action(
        self, key: Key, state: RPQNState
    ) -> tuple[Key, RPQNState, Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action, None

    def _epsilon_greedy_action(
        self, key: Key, state: RPQNState
    ) -> tuple[Key, RPQNState, Array]:

        key, state, random_action, _ = self._random_action(key, state)

        key, state, greedy_action, q_values = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, state, action, q_values

    def _step(self, carry, _, *, policy: Callable) -> tuple[Key, RPQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action, q_values = policy(action_key, state)
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
            value=q_values,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            obs=next_obs,  # type: ignore
            done=done,  # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    def _lambda_backscan(self, carry, transition):
        lambda_return, next_q_value = carry

        target_bootstrap = (
            transition.reward + self.cfg.gamma * (1.0 - transition.done) * next_q_value
        )

        delta = lambda_return - next_q_value
        lambda_return = target_bootstrap + self.cfg.gamma * self.cfg.td_lambda * delta

        lambda_return = (
            1.0 - transition.done
        ) * lambda_return + transition.done * transition.reward

        q_value = jnp.max(transition.value, axis=-1).squeeze(-1)
        return (lambda_return, q_value), lambda_return

    def _update_epoch(self, carry, _):
        key, state, initial_hidden_state, transitions, lambda_targets = carry

        key, permutation_key = jax.random.split(key)
        permutation = jax.random.permutation(permutation_key, self.cfg.num_envs)
        batch = (initial_hidden_state, transitions, lambda_targets)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                [self.cfg.num_minibatches, -1] + list(x.shape[1:]),
            ),
            shuffled_batch,
        )

        (key, state), (loss, q_value) = jax.lax.scan(
            self._update_minibatch, (key, state), xs=minibatches
        )
        return (key, state, initial_hidden_state, transitions, lambda_targets), (
            loss,
            q_value,
        )

    def _update_minibatch(self, carry, minibatch) -> tuple[RPQNState, Array, Array]:
        key, state = carry

        hidden_state, transitions, target = minibatch

        key, memory_key = jax.random.split(key)

        def loss_fn(params):
            _, q_value = self.q_network.apply(
                params,
                observation=transitions.obs,
                mask=transitions.prev_done,
                initial_carry=hidden_state,
                rngs={"memory": memory_key},
            )
            action = jnp.expand_dims(transitions.action, axis=-1)
            q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
            loss = 0.5 * jnp.square(q_value - target).mean()
            return loss, q_value

        (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        state = state.replace(
            params=params,
            optimizer_state=optimizer_state,
        )

        return (key, state), (loss, q_value.mean())

    def _learn(
        self, carry: tuple[Key, RPQNState], _
    ) -> tuple[tuple[Key, RPQNState], dict]:
        key, state = carry

        initial_hidden_state = state.hidden_state
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.num_steps,
        )

        key, memory_key = jax.random.split(key)

        _, final_q_values = self.q_network.apply(
            state.params,
            observation=jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.hidden_state,
            rngs={"memory": memory_key},
        )
        final_q_value = jnp.max(final_q_values, axis=-1).squeeze(-1) * (
            1.0 - state.done
        )

        initial_lambda_return = transitions.reward[-1] + self.cfg.gamma * final_q_value
        _, targets = jax.lax.scan(
            self._lambda_backscan,
            (initial_lambda_return, final_q_value),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, initial_lambda_return[jnp.newaxis]))

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        lambda_targets = jnp.swapaxes(lambda_targets, 0, 1)

        (key, state, _, transitions, _), (loss, q_value) = jax.lax.scan(
            self._update_epoch,
            (key, state, initial_hidden_state, transitions, lambda_targets),
            None,
            self.cfg.update_epochs,
        )

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key) -> tuple[Key, RPQNState, Array, gymnax.EnvState]:
        """
        Initialize environment, network parameters, optimizer, and replay buffer.

        Args:
            key: JAX PRNG key for randomness.

        Returns:
            key: Updated PRNG key after splits.
            state: Initialized RPQNState with params, target_params, optimizer_state, buffer_state.
            obs: Initial observations from vectorized envs.
            env_state: Initial environment state.
        """
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros(self.cfg.num_envs, dtype=bool)
        hidden_state = self.q_network.initialize_carry(obs.shape)
        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            observation=jnp.expand_dims(obs, 1),
            mask=jnp.expand_dims(done, 1),
            initial_carry=hidden_state,
        )
        optimizer_state = self.optimizer.init(params)

        return (
            key,
            RPQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                hidden_state=hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(
        self, key: Key, state: RPQNState, num_steps: int
    ) -> tuple[Key, RPQNState]:
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: Key,
        state: RPQNState,
        num_steps: int,
    ) -> tuple[Key, RPQNState, dict]:
        """
        Run training loop for specified number of steps.

        Args:
            key: JAX PRNG key.
            state: Current RPQNState.
            obs: Current env observations.
            env_state: Current env state.
            num_steps: Total environment steps to train.

        Returns:
            key: Updated PRNG key.
            state: Updated RPQNState after training.
            obs: Latest observations.
            env_state: Latest env state.
            info: Training statistics (loss, rewards, etc.).
        """

        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // (self.cfg.num_steps * self.cfg.num_envs)),
        )
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, -1, 1), transitions)
        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), transitions
        )

        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: RPQNState, num_steps: int) -> tuple[Key, dict]:
        """
        Evaluate current policy for a fixed number of steps without exploration.

        Args:
            key: JAX PRNG key.
            state: RPQNState with trained parameters.
            num_steps: Number of evaluation steps.

        Returns:
            key: Updated PRNG key.
            info: Evaluation metrics (rewards, episode lengths, etc.).
        """
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=bool)
        hidden_state = self.q_network.initialize_carry(obs.shape)

        state = state.replace(
            obs=obs, done=done, hidden_state=hidden_state, env_state=env_state
        )

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
