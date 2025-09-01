from functools import partial
import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import core
from hydra.utils import instantiate
from omegaconf import DictConfig

from memory_rl.loggers import Logger
from memory_rl.networks import RecurrentNetwork, heads
from memory_rl.utils import compute_recurrent_gae as compute_gae, Transition


@chex.dataclass(frozen=True)
class RPPOState:
    step: int
    obs: chex.Array
    done: chex.Array
    env_state: gymnax.EnvState
    actor_hidden_state: chex.Array
    critic_hidden_state: chex.Array
    actor_params: core.FrozenDict[str, chex.ArrayTree]
    critic_params: core.FrozenDict[str, chex.ArrayTree]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class RPPO:
    cfg: DictConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    actor_network: RecurrentNetwork
    critic_network: RecurrentNetwork
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    
    @partial(jax.jit, static_argnums=(0,))
    def init(self, key):
        key, env_key, actor_key, actor_memory_key, critic_key, critic_memory_key = (
            jax.random.split(key, 6)
        )

        env_keys = jax.random.split(env_key, self.cfg.algorithm.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_envs, dtype=bool)
        actor_hidden_state = self.actor_network.initialize_carry(obs.shape)
        critic_hidden_state = self.critic_network.initialize_carry(obs.shape)

        dummy_obs_for_init = jnp.expand_dims(obs, 1)  # (num_envs, 1, obs_dim)
        dummy_mask_for_init = jnp.expand_dims(done, 1)  # (num_envs, 1)

        actor_params = self.actor_network.init(
            {"params": actor_key, "memory": actor_memory_key},
            observation=dummy_obs_for_init,
            mask=dummy_mask_for_init,
            initial_carry=actor_hidden_state,
        )
        critic_params = self.critic_network.init(
            {"params": critic_key, "memory": critic_memory_key},
            observation=dummy_obs_for_init,
            mask=dummy_mask_for_init,
            initial_carry=critic_hidden_state,
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        return (
            key,
            RPPOState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                done=done,  # type: ignore
                actor_hidden_state=actor_hidden_state,  # type: ignore
                critic_hidden_state=critic_hidden_state,  # type: ignore
                env_state=env_state,  # type: ignore
                actor_params=actor_params,  # type: ignore
                critic_params=critic_params,  # type: ignore
                actor_optimizer_state=actor_optimizer_state,  # type: ignore
                critic_optimizer_state=critic_optimizer_state,  # type: ignore
            ),
        )

    def warmup(self, key, state, num_steps):
        """No warmup needed for RPPO"""
        return key, state

    def _step(self, carry: tuple, _):
        key, state = carry

        key, action_key, step_key, actor_memory_key, critic_memory_key = (
            jax.random.split(key, 5)
        )

        actor_h_next, probs = self.actor_network.apply(
            state.actor_params,
            observation=jnp.expand_dims(state.obs, 1),  # (B, 1, F_obs)
            mask=jnp.expand_dims(state.done, 1),  # (B, 1) mask
            initial_carry=state.actor_hidden_state,
            rngs={"memory": actor_memory_key},
        )
        critic_h_next, value = self.critic_network.apply(
            state.critic_params,
            observation=jnp.expand_dims(state.obs, 1),  # (B, 1, F_obs)
            mask=jnp.expand_dims(state.done, 1),  # (B, 1) mask
            initial_carry=state.critic_hidden_state,
            rngs={"memory": critic_memory_key},
        )

        value = value.squeeze((1, -1))
        action = probs.sample(seed=action_key)
        log_prob = probs.log_prob(action)
        action = action.squeeze(1)
        log_prob = log_prob.squeeze(1)

        step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
        next_obs, env_state, reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)
        transition = Transition(
            obs=state.obs,  # type: ignore
            done=state.done,  # type: ignore
            action=action,  # type: ignore
            reward=reward,  # type: ignore
            next_done=next_done,
            info=info,
            log_prob=log_prob,  # type: ignore
            value=value,  # type: ignore
        )

        state = state.replace(
            step=state.step + self.cfg.algorithm.num_envs,
            obs=next_obs,
            env_state=env_state,
            done=next_done,
            actor_hidden_state=actor_h_next,
            critic_hidden_state=critic_h_next,
        )
        return (key, state), transition

    def _actor_loss_fn(
        self, params, actor_key, initial_hidden_state, transitions, advantages
    ):
        _, probs = self.actor_network.apply(
            params,
            observation=transitions.obs,
            mask=transitions.done,
            initial_carry=initial_hidden_state,
            rngs={"memory": actor_key},
        )
        log_probs = probs.log_prob(transitions.action)
        entropy = probs.entropy().mean()
        ratio = jnp.exp(log_probs - transitions.log_prob)
        approx_kl = jnp.mean(transitions.log_prob - log_probs)
        clipfrac = jnp.mean(
            (jnp.abs(ratio - 1.0) > self.cfg.algorithm.clip_coef).astype(jnp.float32)
        )

        actor_loss = -jnp.minimum(
            ratio * advantages,
            jnp.clip(
                ratio,
                1.0 - self.cfg.algorithm.clip_coef,
                1.0 + self.cfg.algorithm.clip_coef,
            )
            * advantages,
        ).mean()
        return actor_loss - self.cfg.algorithm.ent_coef * entropy, (
            entropy,
            approx_kl,
            clipfrac,
        )

    def _critic_loss_fn(
        self, params, critic_key, initial_hidden_state, transitions, returns
    ):
        _, values = self.critic_network.apply(
            params,
            observation=transitions.obs,
            mask=transitions.done,
            initial_carry=initial_hidden_state,
            rngs={"memory": critic_key},
        )
        values = values.squeeze(-1)

        if self.cfg.algorithm.clip_vloss:
            critic_loss = jnp.square(values - returns)
            clipped_value = transitions.value + jnp.clip(
                (values - transitions.value),
                -self.cfg.algorithm.clip_coef,
                self.cfg.algorithm.clip_coef,
            )
            clipped_critic_loss = jnp.square(clipped_value - returns)
            critic_loss = 0.5 * jnp.maximum(critic_loss, clipped_critic_loss).mean()
        else:
            critic_loss = 0.5 * jnp.square(values - returns).mean()

        return critic_loss

    def _update_minibatch(self, carry, minibatch: tuple):
        key, state = carry
        (
            initial_actor_h_mb,
            initial_critic_h_mb,
            transitions,
            advantages,
            returns,
        ) = minibatch

        key, actor_key, critic_key = jax.random.split(key, 3)
        (actor_loss, aux), actor_grads = jax.value_and_grad(
            self._actor_loss_fn, has_aux=True
        )(state.actor_params, actor_key, initial_actor_h_mb, transitions, advantages)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        critic_loss, critic_grads = jax.value_and_grad(self._critic_loss_fn)(
            state.critic_params, critic_key, initial_critic_h_mb, transitions, returns
        )
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return (key, state), (actor_loss, critic_loss, aux)

    def _update_epoch(self, carry: tuple, _):

        (
            key,
            state,
            initial_actor_h_epoch,
            initial_critic_h_epoch,
            transitions,
            advantages,
            returns,
        ) = carry

        key, permutation_key = jax.random.split(key)

        permutation = jax.random.permutation(
            permutation_key, self.cfg.algorithm.num_envs
        )

        batch = (
            initial_actor_h_epoch,
            initial_critic_h_epoch,
            transitions,
            advantages,
            returns,
        )

        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                [self.cfg.algorithm.num_minibatches, -1] + list(x.shape[1:]),
            ),
            shuffled_batch,
        )

        (key, state), (actor_loss, critic_loss, aux) = jax.lax.scan(
            self._update_minibatch,
            (key, state),
            minibatches,
        )

        return (
            key,
            state,
            initial_actor_h_epoch,
            initial_critic_h_epoch,
            transitions,
            advantages,
            returns,
        ), (actor_loss, critic_loss, aux)

    def _update_step(self, carry: tuple, _):

        key, state = carry
        initial_actor_h_rollout = state.actor_hidden_state
        initial_critic_h_rollout = state.critic_hidden_state
        (key, state), transitions = jax.lax.scan(
            self._step,
            (key, state),
            length=self.cfg.algorithm.mode.length,
        )

        key, critic_key = jax.random.split(key)
        _, final_value = self.critic_network.apply(
            state.critic_params,
            observation=jnp.expand_dims(state.obs, 1),
            mask=jnp.expand_dims(state.done, 1),
            initial_carry=state.critic_hidden_state,  # Use the latest critic hidden state from scan
            rngs={"memory": critic_key},
        )

        final_value = final_value.squeeze((1, -1))

        # Compute GAE on time-major data (T, B, ...)
        advantages, returns = compute_gae(
            self.cfg.algorithm.gamma,
            self.cfg.algorithm.gae_lambda,
            transitions,
            final_value,
            state.done,
        )

        # Convert from time_major=True format (T, B, ...) to time_major=False format (B, T, ...)
        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        # Convert advantages and returns from (T, B) to (B, T) format
        advantages = jnp.swapaxes(advantages, 0, 1)
        returns = jnp.swapaxes(returns, 0, 1)

        if self.cfg.algorithm.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        (key, state, *_), (actor_loss, critic_loss, aux) = jax.lax.scan(
            self._update_epoch,
            (
                key,
                state,
                initial_actor_h_rollout,  # (B,H)
                initial_critic_h_rollout,  # (B,H)
                transitions,
                advantages,
                returns,
            ),
            length=self.cfg.algorithm.update_epochs,
        )

        info = transitions.info

        info["losses/actor_loss"] = actor_loss.mean()
        info["losses/critic_loss"] = critic_loss.mean()

        entropy, approx_kl, clipfrac = aux
        info["losses/entropy"] = entropy.mean()
        info["losses/approx_kl"] = approx_kl.mean()
        info["losses/clipfrac"] = clipfrac.mean()

        return (
            key,
            state,
        ), info
        
    @partial(jax.jit, static_argnums=(0, 3), donate_argnums=(2,))
    def train(self, key, state, num_steps):

        (key, state), info = jax.lax.scan(
            self._update_step,
            (key, state),
            length=num_steps
            // (self.cfg.algorithm.num_envs * self.cfg.algorithm.mode.length),
        )

        return key, state, info

    @partial(jax.jit, static_argnums=(0, 3), donate_argnums=(2,))
    def evaluate(self, key, state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.algorithm.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.cfg.algorithm.num_envs, dtype=bool)
        initial_actor_hidden_state = self.actor_network.initialize_carry(obs.shape)

        def step(carry, _):
            key, obs, done, env_state, actor_h_state = carry

            next_actor_h_state, probs = self.actor_network.apply(
                state.actor_params,
                observation=jnp.expand_dims(obs, 1),  # CORRECT: (num_envs, 1, obs_dim)
                mask=jnp.expand_dims(done, 1),  # CORRECT: (num_envs, 1)
                initial_carry=actor_h_state,
            )
            action = jnp.argmax(probs.logits, axis=-1).squeeze(1)  # CORRECT: squeeze(1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.algorithm.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            transition = Transition(
                reward=reward,  # type: ignore
                done=done,  # type: ignore
                info=info,  # type: ignore
            )

            return (key, next_obs, done, env_state, next_actor_h_state), transition

        (key, *_), transitions = jax.lax.scan(
            step,
            (key, obs, done, env_state, initial_actor_hidden_state),
            length=num_steps,
        )

        return key, transitions


def make_rppo(cfg, env, env_params, logger):
    key = jax.random.key(cfg.seed)
    key, reset_key = jax.random.split(key, 2)

    reset_key = jax.random.split(reset_key, cfg.algorithm.num_envs)

    actor = RecurrentNetwork(
        feature_extractor=instantiate(cfg.algorithm.actor.feature_extractor),
        torso=instantiate(cfg.algorithm.actor.torso),
        head=heads.Categorical(
            action_dim=env.action_space(env_params).n,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        ),
    )

    critic = RecurrentNetwork(
        feature_extractor=instantiate(cfg.algorithm.critic.feature_extractor),
        torso=instantiate(cfg.algorithm.critic.torso),
        head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
    )

    if cfg.algorithm.anneal_lr:
        num_updates_per_epoch = cfg.total_timesteps // (
            cfg.algorithm.num_envs * cfg.algorithm.mode.length
        )
        num_updates = (
            num_updates_per_epoch
            * cfg.algorithm.update_epochs
            * cfg.algorithm.num_minibatches
        )
        learning_rate = optax.linear_schedule(
            init_value=cfg.algorithm.learning_rate,
            end_value=0.0,
            transition_steps=num_updates,
        )
    else:
        learning_rate = cfg.algorithm.learning_rate

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.algorithm.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.algorithm.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    return RPPO(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor_network=actor,
        critic_network=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
