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
class SoftPQNState:
    step: int
    obs: chex.Array
    env_state: gymnax.EnvState
    params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    log_alpha: chex.Array
    alpha_opt_state: optax.OptState
    log_beta: chex.Array
    beta_opt_state: optax.OptState


@chex.dataclass(frozen=True)
class SoftPQN:
    """
    Deep Q-Network (SoftPQN) reinforcement learning algorithm (soft/Boltzmann variant).
    """

    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    q_network: Network
    optimizer: optax.GradientTransformation

    alpha_optimizer: optax.GradientTransformation
    beta_optimizer: optax.GradientTransformation

    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: chex.PRNGKey, state: SoftPQNState
    ) -> tuple[chex.PRNGKey, chex.Array, chex.Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        action = jnp.argmax(q_values, axis=-1)
        return key, action, q_values

    def _boltzmann_action(
        self, key: chex.PRNGKey, state: SoftPQNState
    ) -> tuple[chex.PRNGKey, chex.Array, chex.Array]:
        q_values = self.q_network.apply(state.params, state.obs)
        alpha = jnp.exp(state.log_alpha)
        logits = q_values / jnp.clip(alpha, a_min=1e-6)  # stable temperature division
        key, sk = jax.random.split(key)
        action = jax.random.categorical(sk, logits=logits, axis=-1)
        return key, action, q_values

    def _step(self, carry, _, *, policy: Callable) -> tuple[chex.PRNGKey, SoftPQNState]:
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, action, q_values = policy(action_key, state)
        num_envs = state.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.obs,        # type: ignore
            action=action,        # type: ignore
            reward=reward,        # type: ignore
            done=done,            # type: ignore
            next_obs=next_obs,    # type: ignore
            info=info,            # type: ignore
            value=q_values,       # type: ignore (stores Q(s,·) for entropy/KL)
        )

        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,         # type: ignore
            env_state=env_state,  # type: ignore
        )
        return (key, state), transition

    @staticmethod
    def _policy_stats_from_q(q_values: chex.Array, alpha: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Return (probs, log_probs, entropy, soft_state_value)."""
        logits = q_values / jnp.clip(alpha, a_min=1e-6)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)                          # H(π)
        # soft_v = jnp.sum(probs * (q_values - alpha * log_probs), axis=-1)
        soft_v = alpha * jax.nn.logsumexp(q_values / alpha, axis=-1)
        return probs, log_probs, entropy, soft_v

    @staticmethod
    def _kl_to_uniform(log_probs: chex.Array) -> chex.Array:
        a_dim = log_probs.shape[-1]
        probs = jnp.exp(log_probs)
        kl = jnp.sum(probs * log_probs, axis=-1) + jnp.log(jnp.asarray(a_dim, dtype=log_probs.dtype))
        return kl

    def _lambda_backscan(self, carry, transition, *, alpha: chex.Array):
        lambda_return, next_soft_v = carry

        target_bootstrap = (
            transition.reward
            + self.cfg.algorithm.gamma * (1.0 - transition.done) * next_soft_v
        )

        delta = lambda_return - next_soft_v
        lambda_return = (
            target_bootstrap
            + self.cfg.algorithm.gamma * self.cfg.algorithm.td_lambda * delta
        )

        lambda_return = (
            1.0 - transition.done
        ) * lambda_return + transition.done * transition.reward

        _, logp, _, soft_v = self._policy_stats_from_q(transition.value, alpha)
        soft_v = soft_v
        return (lambda_return, soft_v), lambda_return

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

        (key, state), (loss, q_value, kl, entropy, alpha_loss, beta_loss) = jax.lax.scan(
            self._update_minibatch, (key, state), xs=(minibatches, targets)
        )
        return (key, state, transitions, lambda_targets), (loss, q_value, kl, entropy, alpha_loss, beta_loss)

    def _update_minibatch(self, carry, xs):
        key, state = carry
        minibatch, target = xs

        alpha = jnp.exp(state.log_alpha)
        beta = jnp.exp(state.log_beta)

        def loss_fn(params):
            q_values = self.q_network.apply(params, minibatch.obs)
            log_probs = jax.nn.log_softmax(q_values / jnp.clip(alpha, 1e-6), axis=-1)
            probs = jnp.exp(log_probs)

            entropy = -jnp.sum(probs * log_probs, axis=-1)
            kl = self._kl_to_uniform(log_probs)

            action = jnp.expand_dims(minibatch.action, axis=-1)
            q_sa = jnp.take_along_axis(q_values, action, axis=-1).squeeze(-1)
            td_loss = 0.5 * jnp.square(q_sa - target).mean()

            total_loss = td_loss + kl_penalty
            return total_loss, (q_sa.mean(), kl.mean(), entropy.mean(), td_loss)

        (total_loss, (q_mean, kl_mean, ent_mean, td_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)

        target_entropy = getattr(self.cfg.algorithm, "target_entropy", None)
        if target_entropy is None:
            action_dim = self.env.action_space(self.env_params).n
            target_entropy = 0.98 * jnp.log(jnp.asarray(action_dim, dtype=jnp.float32))

        alpha_loss = jnp.exp(state.log_alpha) * (jax.lax.stop_gradient(ent_mean) - target_entropy)
        alpha_grads = jax.grad(lambda la: jnp.exp(la) * (jax.lax.stop_gradient(ent_mean) - target_entropy))(state.log_alpha)
        alpha_updates, alpha_opt_state = self.alpha_optimizer.update(alpha_grads, state.alpha_opt_state, state.log_alpha)
        log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

        target_kl = getattr(self.cfg.algorithm, "target_kl", 0.05 * target_entropy)  # sensible default
        beta_loss = -jnp.exp(state.log_beta) * (jax.lax.stop_gradient(kl_mean) - target_kl)
        beta_grads = jax.grad(lambda lb: -jnp.exp(lb) * (jax.lax.stop_gradient(kl_mean) - target_kl))(state.log_beta)
        beta_updates, beta_opt_state = self.beta_optimizer.update(beta_grads, state.beta_opt_state, state.log_beta)
        log_beta = optax.apply_updates(state.log_beta, beta_updates)

        state = state.replace(
            params=params,
            optimizer_state=optimizer_state,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            log_beta=log_beta,
            beta_opt_state=beta_opt_state,
        )

        return (key, state), (total_loss, q_mean, kl_mean, ent_mean, alpha_loss, beta_loss)

    def _learn(
        self, carry: tuple[chex.PRNGKey, SoftPQNState], _
    ) -> tuple[tuple[chex.PRNGKey, SoftPQNState], dict]:

        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._boltzmann_action),
            carry,
            length=self.cfg.algorithm.num_steps,
        )

        alpha = jnp.exp(state.log_alpha)

        final_next_obs = transitions.next_obs[-1]
        final_done = transitions.done[-1]
        final_reward = transitions.reward[-1]

        final_q_values = self.q_network.apply(state.params, final_next_obs)
        _, _, final_entropy, final_soft_v = self._policy_stats_from_q(final_q_values, alpha)
        final_soft_v = final_soft_v * (1.0 - final_done)

        lambda_returns = final_reward + self.cfg.algorithm.gamma * final_soft_v

        (_, _), targets = jax.lax.scan(
            partial(self._lambda_backscan, alpha=alpha),
            (lambda_returns, final_soft_v),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        (key, state, transitions, _), (loss, q_value, kl, entropy, alpha_loss, beta_loss) = jax.lax.scan(
            self._update_epoch,
            (key, state, transitions, lambda_targets),
            None,
            self.cfg.algorithm.update_epochs,
        )


        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value
        transitions.info["losses/kl"] = kl
        transitions.info["losses/entropy"] = entropy
        transitions.info["losses/alpha_loss"] = alpha_loss
        transitions.info["losses/beta_loss"] = beta_loss

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["key"])
    def init(self, key) -> tuple[chex.PRNGKey, SoftPQNState, chex.Array, gymnax.EnvState]:
        key, env_key, q_key, a_key, b_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        params = self.q_network.init(q_key, obs)
        optimizer_state = self.optimizer.init(params)

        log_alpha = jnp.log(jnp.asarray(getattr(self.cfg.algorithm, "init_alpha", 1.0), dtype=jnp.float32))
        log_beta = jnp.log(jnp.asarray(getattr(self.cfg.algorithm, "init_beta", 0.1), dtype=jnp.float32))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        beta_opt_state = self.beta_optimizer.init(log_beta)

        return (
            key,
            SoftPQNState(
                step=0,                 # type: ignore
                obs=obs,                # type: ignore
                env_state=env_state,    # type: ignore
                params=params,          # type: ignore
                optimizer_state=optimizer_state,  # type: ignore
                log_alpha=log_alpha,    # type: ignore
                alpha_opt_state=alpha_opt_state,  # type: ignore
                log_beta=log_beta,      # type: ignore
                beta_opt_state=beta_opt_state,    # type: ignore
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key"])
    def warmup(self, key: chex.PRNGKey, state: SoftPQNState, num_steps: int) -> tuple[chex.PRNGKey, PQNState]:
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"])
    def train(
        self,
        key: chex.PRNGKey,
        state: SoftPQNState,
        num_steps: int,
    ) -> tuple[chex.PRNGKey, SoftPQNState, dict]:
        (key, state), transitions = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // (self.cfg.algorithm.num_steps * self.cfg.algorithm.num_envs)),
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"], donate_argnames=["key", "state"])
    def evaluate(
        self, key: chex.PRNGKey, state: SoftPQNState, num_steps: int
    ) -> tuple[chex.PRNGKey, dict]:
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
        return key, transitions.replace(obs=None, next_obs=None)


def make_soft_pqn(cfg, env, env_params) -> SoftPQN:
    """
    Factory function to construct a SoftPQN agent from Args.
    """
    action_dim = env.action_space(env_params).n

    q_network = Network(
        feature_extractor=instantiate(cfg.algorithm.feature_extractor),
        torso=instantiate(cfg.algorithm.torso),
        head=heads.DiscreteQNetwork(action_dim=action_dim),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.algorithm.max_grad_norm),
        optax.adam(learning_rate=cfg.algorithm.learning_rate),
    )

    alpha_optimizer = optax.adam(learning_rate=getattr(cfg.algorithm, "alpha_lr", 3e-4))
    beta_optimizer = optax.adam(learning_rate=getattr(cfg.algorithm, "beta_lr", 3e-4))

    epsilon_schedule = optax.constant_schedule(0.0)

    return SoftPQN(
        cfg=cfg,                       # type: ignore
        env=env,                       # type: ignore
        env_params=env_params,         # type: ignore
        q_network=q_network,           # type: ignore
        optimizer=optimizer,           # type: ignore
        alpha_optimizer=alpha_optimizer,  # type: ignore
        beta_optimizer=beta_optimizer,    # type: ignore
        epsilon_schedule=epsilon_schedule, # type: ignore
    )

