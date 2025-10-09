from functools import partial
from typing import Any, Callable

import chex
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core

from memory_rl.networks import Network
from memory_rl.utils import Transition


def boltzmann_policy(q, tau: float):
    logits = q / tau
    pi = jax.nn.softmax(logits, axis=-1)
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    entropy = -(pi * log_pi).sum(axis=-1)
    return pi, log_pi, entropy


@chex.dataclass(frozen=True)
class SoftPQNConfig:
    name: str
    learning_rate: float
    num_envs: int
    num_eval_envs: int
    num_steps: int
    anneal_lr: bool
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    learning_starts: int
    actor: Any
    critic: Any

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@chex.dataclass(frozen=True)
class SoftPQNState:
    """
    Immutable container for training state of SoftPQN algorithm.
    """

    step: int
    obs: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    alpha: core.FrozenDict[str, Any]
    beta: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    beta_optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class SoftPQN:
    """
    Deep Q-Network (SoftPQN) reinforcement learning algorithm.
    """

    cfg: SoftPQNConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    alpha_network: Network
    beta_network: Network
    optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    beta_optimizer: optax.GradientTransformation
    target_entropy: float
    target_kl: float

    def _deterministic_action(
        self, key: chex.PRNGKey, state: SoftPQNState
    ) -> tuple[chex.PRNGKey, chex.Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action, q_values, None

    def _stochastic_action(self, key: chex.PRNGKey, state: SoftPQNState):
        num_envs = state.obs.shape[0]
        key, policy_key = jax.random.split(key)
        policy_keys = jax.random.split(policy_key, num_envs)

        q_values = self.q_network.apply(state.params, state.obs)
        log_alpha = self.alpha_network.apply(state.alpha)
        tau = jnp.exp(log_alpha)
        _, log_pi, _ = boltzmann_policy(q_values, tau)
        action = jax.vmap(lambda logits, key: jax.random.categorical(key, logits))(
            log_pi, policy_keys
        )
        return key, action, q_values, log_pi

    def _step(self, carry, _, *, policy: Callable) -> tuple[chex.PRNGKey, SoftPQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action, q_values, log_prob = policy(action_key, state)
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
            log_prob=log_prob,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,  # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    def _lambda_backscan(self, carry, transition, tau):
        lambda_return, next_soft_value = carry

        target_bootstrap = (
            transition.reward
            + self.cfg.algorithm.gamma * (1.0 - transition.done) * next_soft_value
        )

        delta = lambda_return - next_soft_value
        lambda_return = (
            target_bootstrap
            + self.cfg.algorithm.gamma * self.cfg.algorithm.td_lambda * delta
        )

        lambda_return = (
            1.0 - transition.done
        ) * lambda_return + transition.done * transition.reward

        soft_value = tau * jax.nn.logsumexp(transition.value / tau, axis=-1)
        return (lambda_return, soft_value), lambda_return

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

        (key, state), aux = jax.lax.scan(
            self._update_minibatch, (key, state), xs=(minibatches, targets)
        )
        return (key, state, transitions, lambda_targets), aux

    def _update_minibatch(
        self, carry, xs
    ) -> tuple[SoftPQNState, chex.Array, chex.Array]:
        key, state = carry
        minibatch, target = xs

        log_alpha = self.alpha_network.apply(state.alpha)
        tau = jnp.exp(log_alpha)

        log_beta = self.beta_network.apply(state.beta)
        beta = jnp.exp(log_beta)

        log_pi_old = minibatch.log_prob
        pi_old = jnp.exp(log_pi_old)

        def loss_fn(params):
            q_values = self.q_network.apply(
                params,
                minibatch.obs,
            )
            _, log_pi, _ = boltzmann_policy(q_values, tau)
            action = jnp.expand_dims(minibatch.action, axis=-1)
            q_value = jnp.take_along_axis(q_values, action, axis=-1).squeeze(-1)
            td_error = optax.l2_loss(q_value, target).mean()

            kl = jnp.sum(pi_old * (log_pi_old - log_pi), axis=-1).mean()

            loss = td_error + beta * kl
            return loss, q_value.mean()

        (loss, q_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        q_values = self.q_network.apply(params, minibatch.obs)

        def alpha_loss_fn(params):
            log_alpha = self.alpha_network.apply(params)
            tau = jnp.exp(log_alpha)

            _, _, entropy = boltzmann_policy(q_values, tau)

            loss = tau * jax.lax.stop_gradient(entropy - self.target_entropy).mean()
            return loss, jnp.mean(entropy)

        _, log_pi, _ = boltzmann_policy(q_values, tau)
        kl = jnp.sum(pi_old * (log_pi_old - log_pi), axis=-1).mean()

        def beta_loss_fn(params):
            log_beta = self.beta_network.apply(params)
            beta = jnp.exp(log_beta)

            loss = beta * (self.target_kl - kl).mean()
            return loss

        (alpha_loss, entropy), alpha_grads = jax.value_and_grad(
            alpha_loss_fn, has_aux=True
        )(state.alpha)
        alpha_updates, alpha_optimizer_state = self.alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state, state.alpha
        )
        alpha = optax.apply_updates(state.alpha, alpha_updates)

        beta_loss, beta_grads = jax.value_and_grad(beta_loss_fn, has_aux=False)(
            state.beta
        )
        beta_updates, beta_optimizer_state = self.beta_optimizer.update(
            beta_grads, state.beta_optimizer_state, state.beta
        )
        beta = optax.apply_updates(state.beta, beta_updates)

        state = state.replace(
            params=params,
            alpha=alpha,
            beta=beta,
            optimizer_state=optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            beta_optimizer_state=beta_optimizer_state,
        )

        return (key, state), (loss, alpha_loss, beta_loss, q_value, entropy, kl)

    def _learn(
        self, carry: tuple[chex.PRNGKey, SoftPQNState], _
    ) -> tuple[tuple[chex.PRNGKey, SoftPQNState], dict]:

        key, state = carry

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            carry,
            length=self.cfg.algorithm.num_steps,
        )

        log_alpha = self.alpha_network.apply(state.alpha)
        tau = jnp.exp(log_alpha)

        final_next_obs = transitions.next_obs[-1]
        final_done = transitions.done[-1]
        final_reward = transitions.reward[-1]

        final_q_values = self.q_network.apply(state.params, final_next_obs)
        final_soft_value = (
            (1 - final_done) * tau * jax.nn.logsumexp(final_q_values / tau, axis=-1)
        )

        lambda_returns = final_reward + self.cfg.algorithm.gamma * final_soft_value
        _, targets = jax.lax.scan(
            partial(self._lambda_backscan, tau=tau),
            (lambda_returns, final_soft_value),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        (key, state, transitions, _), aux = jax.lax.scan(
            self._update_epoch,
            (key, state, transitions, lambda_targets),
            None,
            self.cfg.algorithm.update_epochs,
        )

        loss, alpha_loss, beta_loss, q_value, entropy, kl = aux
        transitions.info["losses/loss"] = loss
        transitions.info["losses/alpha_loss"] = alpha_loss
        transitions.info["losses/beta_loss"] = beta_loss
        transitions.info["losses/q_value"] = q_value
        transitions.info["losses/entropy"] = entropy
        transitions.info["losses/target_entropy"] = self.target_entropy
        transitions.info["losses/kl"] = kl
        transitions.info["losses/target_kl"] = self.target_kl

        return (key, state), transitions

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(
        self, key
    ) -> tuple[chex.PRNGKey, SoftPQNState, chex.Array, gymnax.EnvState]:
        """
        Initialize environment, network parameters, optimizer, and replay buffer.

        Args:
            key: JAX PRNG key for randomness.

        Returns:
            key: Updated PRNG key after splits.
            state: Initialized SoftPQNState with params, target_params, optimizer_state, buffer_state.
            obs: Initial observations from vectorized envs.
            env_state: Initial environment state.
        """
        key, env_key, q_key, alpha_key, beta_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        params = self.q_network.init(q_key, obs)
        alpha = self.alpha_network.init(alpha_key)
        beta = self.beta_network.init(beta_key)

        optimizer_state = self.optimizer.init(params)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha)
        beta_optimizer_state = self.beta_optimizer.init(beta)

        return (
            key,
            SoftPQNState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                params=params,  # type: ignore
                alpha=alpha,  # type: ignore
                beta=beta,  # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                alpha_optimizer_state=alpha_optimizer_state,  # type: ignore
                beta_optimizer_state=beta_optimizer_state,  # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key"])
    def warmup(
        self, key: chex.PRNGKey, state: SoftPQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, SoftPQNState]:
        return key, state

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def train(
        self,
        key: chex.PRNGKey,
        state: SoftPQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, SoftPQNState, dict]:
        """
        Run training loop for specified number of steps.

        Args:
            key: JAX PRNG key.
            state: Current SoftPQNState.
            obs: Current env observations.
            env_state: Current env state.
            num_steps: Total environment steps to train.

        Returns:
            key: Updated PRNG key.
            state: Updated SoftPQNState after training.
            obs: Latest observations.
            env_state: Latest env state.
            info: Training statistics (loss, rewards, etc.).
        """
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(
                num_steps
                // (self.cfg.algorithm.num_steps * self.cfg.algorithm.num_envs)
            ),
        )
        return key, state, transitions

    @partial(
        jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"]
    )
    def evaluate(
        self, key: chex.PRNGKey, state: SoftPQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:
        """
        Evaluate current policy for a fixed number of steps without exploration.

        Args:
            key: JAX PRNG key.
            state: SoftPQNState with trained parameters.
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
            partial(self._step, policy=self._deterministic_action),
            (key, state),
            length=num_steps,
        )

        return key, transitions
