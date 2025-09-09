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
class PQNState:
    """
    Immutable container for training state of PQN algorithm.
    """

    step: int
    obs: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class PQN:
    """
    Deep Q-Network (PQN) reinforcement learning algorithm.
    """

    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: chex.PRNGKey, state: PQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action, q_values

    def _random_action(
        self, key: chex.PRNGKey, state: PQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.algorithm.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, action, None

    def _epsilon_greedy_action(
        self, key: chex.PRNGKey, state: PQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:

        key, random_action, _ = self._random_action(key, state)

        key, greedy_action, q_values = self._greedy_action(key, state)

        key, sample_key = jax.random.split(key)
        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return key, action, q_values

    def _step(self, carry, _, *, policy: Callable) -> tuple[chex.PRNGKey, PQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action, q_values = policy(action_key, state)
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
            next_obs=next_obs,  # type: ignore
            info=info,  # type: ignore
            value=q_values,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    def _lambda_backscan(self, carry, transition):
        lambda_return, next_q_value = carry

        target_bootstrap = (
            transition.reward
            + self.cfg.algorithm.gamma * (1.0 - transition.done) * next_q_value
        )

        delta = lambda_return - next_q_value
        lambda_return = (
            target_bootstrap
            + self.cfg.algorithm.gamma * self.cfg.algorithm.td_lambda * delta
        )

        lambda_return = (
            1.0 - transition.done
        ) * lambda_return + transition.done * transition.reward

        q_value = jnp.max(transition.value, axis=-1)
        return (lambda_return, q_value), lambda_return

    def _preprocess_transition(self, key, x):
        x = x.reshape(-1, *x.shape[2:])
        x = jax.random.permutation(key, x)
        x = x.reshape(self.cfg.algorithm.num_minibatches, -1, *x.shape[1:])
        return x

    def _update_epoch(self, carry, _):
        key, state, transitions, lambda_targets = carry

        key, split_key = jax.random.split(key)
        minibatches = jax.tree_util.tree_map(
            lambda x: self._preprocess_transition(split_key, x), transitions
        )
        targets = self._preprocess_transition(split_key, lambda_targets)

        (key, state), (loss, q_value) = jax.lax.scan(
            self._update_minibatch, (key, state), xs=(minibatches, targets)
        )
        return (key, state, transitions, lambda_targets), (loss, q_value)

    def _update_minibatch(self, carry, xs) -> tuple[PQNState, chex.Array, chex.Array]:
        key, state = carry
        minibatch, target = xs

        def loss_fn(params):
            q_value = self.q_network.apply(
                params,
                minibatch.obs,
            )
            action = jnp.expand_dims(minibatch.action, axis=-1)
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
        self, carry: tuple[chex.PRNGKey, PQNState], _
    ) -> tuple[tuple[chex.PRNGKey, PQNState], dict]:

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.algorithm.num_steps,
        )

        final_next_obs = transitions.next_obs[-1]
        final_done = transitions.done[-1]
        final_reward = transitions.reward[-1]

        final_q_values = self.q_network.apply(state.params, final_next_obs)
        final_q_value = jnp.max(final_q_values, axis=-1) * (1.0 - final_done)

        lambda_returns = final_reward + self.cfg.algorithm.gamma * final_q_value
        _, targets = jax.lax.scan(
            self._lambda_backscan,
            (lambda_returns, final_q_value),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        (key, state, transitions, _), (loss, q_value) = jax.lax.scan(
            self._update_epoch,
            (key, state, transitions, lambda_targets),
            None,
            self.cfg.algorithm.update_epochs,
        )

        info = transitions.info
        info["losses/loss"] = loss
        info["losses/q_value"] = q_value

        return (key, state), info

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key) -> tuple[chex.PRNGKey, PQNState, chex.Array, gymnax.EnvState]:
        """
        Initialize environment, network parameters, optimizer, and replay buffer.

        Args:
            key: JAX PRNG key for randomness.

        Returns:
            key: Updated PRNG key after splits.
            state: Initialized PQNState with params, target_params, optimizer_state, buffer_state.
            obs: Initial observations from vectorized envs.
            env_state: Initial environment state.
        """
        key, env_key, q_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        return (
            key,
            PQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key"])
    def warmup(
        self, key: chex.PRNGKey, state: PQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, PQNState]:
        return key, state

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def train(
        self,
        key: chex.PRNGKey,
        state: PQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, PQNState, dict]:
        """
        Run training loop for specified number of steps.

        Args:
            key: JAX PRNG key.
            state: Current PQNState.
            obs: Current env observations.
            env_state: Current env state.
            num_steps: Total environment steps to train.

        Returns:
            key: Updated PRNG key.
            state: Updated PQNState after training.
            obs: Latest observations.
            env_state: Latest env state.
            info: Training statistics (loss, rewards, etc.).
        """
        (key, state), info = jax.lax.scan(
            self._learn,
            (key, state),
            length=(
                num_steps
                // (self.cfg.algorithm.num_steps * self.cfg.algorithm.num_envs)
            ),
        )
        return key, state, info

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def evaluate(
        self, key: chex.PRNGKey, state: PQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:
        """
        Evaluate current policy for a fixed number of steps without exploration.

        Args:
            key: JAX PRNG key.
            state: PQNState with trained parameters.
            num_steps: Number of evaluation steps.

        Returns:
            key: Updated PRNG key.
            info: Evaluation metrics (rewards, episode lengths, etc.).
        """
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        state = state.replace(obs=obs, env_state=env_state)

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions


def make_pqn(cfg, env, env_params) -> PQN:
    """
    Factory function to construct a PQN agent from Args.

    Args:
        args: Experiment configuration.

    Returns:
        An initialized PQN instance ready for training.
    """

    q_network = Network(
        feature_extractor=instantiate(cfg.algorithm.feature_extractor),
        torso=instantiate(cfg.algorithm.torso),
        head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.algorithm.max_grad_norm),
        optax.adam(learning_rate=cfg.algorithm.learning_rate),
    )

    epsilon_schedule = optax.linear_schedule(
        cfg.algorithm.start_e,
        cfg.algorithm.end_e,
        int(cfg.algorithm.exploration_fraction * cfg.total_timesteps),
    )
    return PQN(
        cfg=cfg,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
