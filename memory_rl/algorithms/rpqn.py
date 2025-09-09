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
from memory_rl.utils import Transition


@chex.dataclass(frozen=True)
class RPQNState:
    """
    Immutable container for training state of RPQN algorithm.
    """

    step: int
    obs: chex.Array
    done: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    hidden_state: chex.Array
    optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class RPQN:
    """
    Deep Q-Network (RPQN) reinforcement learning algorithm.
    """

    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: RecurrentNetwork
    optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: chex.PRNGKey, state: RPQNState
    ) -> tuple[chex.PRNGKey, RPQNState, chex.Array]:
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
        self, key: chex.PRNGKey, state: RPQNState
    ) -> tuple[chex.PRNGKey, RPQNState, chex.Array]:
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.algorithm.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action, None

    def _epsilon_greedy_action(
        self, key: chex.PRNGKey, state: RPQNState
    ) -> tuple[chex.PRNGKey, RPQNState, chex.Array]:

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

    def _step(self, carry, _, *, policy: Callable) -> tuple[chex.PRNGKey, RPQNState]:
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
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,  # type: ignore
            done=done,  # type: ignore
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
            1.0 - transition.one
        ) * lambda_return + transition.done * transition.reward

        q_value = jnp.max(transition.value, axis=-1).squeeze(-1)
        return (lambda_return, q_value), lambda_return

    def _update_epoch(self, carry, _):
        key, state, initial_hidden_state, transitions, lambda_targets = carry

        key, permutation_key = jax.random.split(key)
        permutation = jax.random.permutation(
            permutation_key, self.cfg.algorithm.num_envs
        )
        batch = (initial_hidden_state, transitions, lambda_targets)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                [self.cfg.algorithm.num_minibatches, -1] + list(x.shape[1:]),
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

    def _update_minibatch(
        self, carry, minibatch
    ) -> tuple[RPQNState, chex.Array, chex.Array]:
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

    def _update_step(
        self, carry: tuple[chex.PRNGKey, RPQNState], _
    ) -> tuple[tuple[chex.PRNGKey, RPQNState], dict]:
        key, state = carry

        initial_hidden_state = state.hidden_state
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            (key, state),
            length=self.cfg.algorithm.mode.length,
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

        initial_lambda_return = (
            transitions.reward[-1] + self.cfg.algorithm.gamma * final_q_value
        )
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
            self.cfg.algorithm.update_epochs,
        )

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value

        return (key, state), transitions

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key) -> tuple[chex.PRNGKey, RPQNState, chex.Array, gymnax.EnvState]:
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
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_envs, dtype=bool)
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

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key"])
    def warmup(
        self, key: chex.PRNGKey, state: RPQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, RPQNState]:
        return key, state

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def train(
        self,
        key: chex.PRNGKey,
        state: RPQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, RPQNState, dict]:
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
            self._update_step,
            (key, state),
            length=(
                num_steps
                // (self.cfg.algorithm.mode.length * self.cfg.algorithm.num_envs)
            ),
        )
        return key, state, transitions

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def evaluate(
        self, key: chex.PRNGKey, state: RPQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:
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
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_eval_envs, dtype=bool)
        hidden_state = self.q_network.initialize_carry(obs.shape)

        state = state.replace(
            obs=obs, done=done, hidden_state=hidden_state, env_state=env_state
        )

        (key, *_), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions


def make_rpqn(cfg, env, env_params) -> RPQN:
    """
    Factory function to construct a RPQN agent from Args.

    Args:
        args: Experiment configuration.

    Returns:
        An initialized RPQN instance ready for training.
    """

    q_network = RecurrentNetwork(
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
    return RPQN(
        cfg=cfg,  # type: ignore
        env=env,  # type: ignore
        env_params=env_params,  # type: ignore
        q_network=q_network,  # type: ignore
        optimizer=optimizer,  # type: ignore
        epsilon_schedule=epsilon_schedule,  # type: ignore
    )
