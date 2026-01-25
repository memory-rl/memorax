"""R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning.

This implements R2D2 as described in https://openreview.net/pdf?id=r1lyTjAqYX.

Key features:
- Recurrent Q-network for partial observability
- Prioritized experience replay with episode-aware sampling
- N-step returns for multi-step bootstrapping
- Burn-in sequences to warm up recurrent hidden state
- Importance sampling weights to correct for priority bias
"""

from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.buffers import compute_importance_weights
from memorax.networks.sequence_models.utils import (
    add_feature_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import Timestep, Transition, periodic_incremental_update
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
)


@struct.dataclass(frozen=True)
class R2D2Config:
    """Configuration for R2D2 algorithm.

    Attributes:
        name: Name identifier for the algorithm.
        learning_rate: Learning rate for the optimizer.
        num_envs: Number of parallel training environments.
        num_eval_envs: Number of parallel evaluation environments.
        buffer_size: Maximum size of the replay buffer in timesteps.
        gamma: Discount factor for future rewards.
        tau: Soft update coefficient for target network.
        target_network_frequency: Steps between target network updates.
        batch_size: Number of sequences to sample per update.
        start_e: Initial epsilon for exploration.
        end_e: Final epsilon for exploration.
        exploration_fraction: Fraction of training for epsilon decay.
        learning_starts: Number of steps before learning begins.
        train_frequency: Steps between training updates.
        burn_in_length: Number of timesteps to burn-in recurrent state.
        sequence_length: Total sequence length to sample (includes burn-in).
        n_step: Number of steps for n-step returns.
        priority_exponent: Priority exponent (alpha) for PER. 0=uniform, 1=full prioritization.
        importance_sampling_exponent: Initial beta for importance sampling. Annealed to 1.0.
        double: Whether to use double Q-learning for target computation.
    """

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
    burn_in_length: int = 10
    sequence_length: int = 80
    n_step: int = 5
    priority_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6
    double: bool = True


@struct.dataclass(frozen=True)
class R2D2State:
    """State for R2D2 algorithm.

    Attributes:
        step: Current training step.
        timestep: Current environment timestep.
        hidden_state: Current recurrent hidden state.
        env_state: Current environment state.
        params: Q-network parameters.
        target_params: Target Q-network parameters.
        optimizer_state: Optimizer state.
        buffer_state: Replay buffer state.
    """

    step: int
    timestep: Timestep
    hidden_state: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


def compute_n_step_returns(
    rewards: Array,
    dones: Array,
    next_q_values: Array,
    n_step: int,
    gamma: float,
) -> Array:
    """Compute n-step returns with proper episode boundary handling.

    Computes: R_t + gamma*R_{t+1} + ... + gamma^{n-1}*R_{t+n-1} + gamma^n * Q(s_{t+n})

    At episode boundaries (done=True), bootstrapping stops and remaining
    rewards are not included.

    Args:
        rewards: Rewards of shape [batch, seq_len] where seq_len >= n_step.
        dones: Done flags of shape [batch, seq_len].
        next_q_values: Q-values for bootstrapping of shape [batch, seq_len].
        n_step: Number of steps for returns computation.
        gamma: Discount factor.

    Returns:
        N-step returns of shape [batch, seq_len - n_step + 1].
    """
    batch_size, seq_len = rewards.shape
    num_targets = seq_len - n_step + 1

    def compute_single_target(start_idx):
        """Compute n-step return starting from start_idx."""
        # Accumulate discounted rewards
        n_step_return = jnp.zeros(batch_size)
        discount = 1.0
        not_done_yet = jnp.ones(batch_size)

        for i in range(n_step):
            idx = start_idx + i
            # Only add reward if episode hasn't ended
            n_step_return = n_step_return + discount * rewards[:, idx] * not_done_yet
            # Update discount for next step
            discount = discount * gamma
            # Track if episode ended
            not_done_yet = not_done_yet * (1.0 - dones[:, idx])

        # Bootstrap with Q-value at n steps ahead (if episode hasn't ended)
        bootstrap_idx = start_idx + n_step - 1
        n_step_return = n_step_return + discount * next_q_values[:, bootstrap_idx] * not_done_yet

        return n_step_return

    # Compute targets for each starting position
    targets = jax.vmap(compute_single_target)(jnp.arange(num_targets))
    # Transpose from [num_targets, batch] to [batch, num_targets]
    targets = targets.T

    return targets


@struct.dataclass(frozen=True)
class R2D2:
    """R2D2 algorithm implementation.

    R2D2 extends DQN with:
    - Recurrent Q-network for handling partial observability
    - Prioritized experience replay with episode-aware sampling
    - N-step returns for faster credit assignment
    - Burn-in sequences to properly initialize recurrent state
    - Double Q-learning for reduced overestimation
    """

    cfg: R2D2Config
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule
    beta_schedule: optax.Schedule

    def _greedy_action(self, key: Key, state: R2D2State) -> tuple[Key, R2D2State, Array]:
        """Select greedy action based on Q-values."""
        key, memory_key = jax.random.split(key)
        timestep = state.timestep.to_sequence()
        hidden_state, (q_values, _) = self.q_network.apply(
            state.params,
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            state.hidden_state,
            rngs={"memory": memory_key},
        )
        action = jnp.argmax(q_values, axis=-1)
        action = remove_time_axis(action)
        state = state.replace(hidden_state=hidden_state)
        return key, state, action

    def _random_action(self, key: Key, state: R2D2State) -> tuple[Key, R2D2State, Array]:
        """Select random action for exploration."""
        key, action_key = jax.random.split(key)
        action_key = jax.random.split(action_key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return key, state, action

    def _epsilon_greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[Key, R2D2State, Array]:
        """Select action using epsilon-greedy policy."""
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
    ) -> tuple[Key, R2D2State]:
        """Execute one environment step."""
        key, state = carry

        key, action_key, step_key = jax.random.split(key, 3)
        key, state, action = policy(action_key, state)
        num_envs = state.timestep.obs.shape[0]
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        transition = Transition(
            obs=state.timestep.obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info,
            prev_done=state.timestep.done,
        )

        buffer_state = state.buffer_state
        if write_to_buffer:
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
            buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return (key, state), transition

    def _update(self, key: Key, state: R2D2State) -> tuple[R2D2State, Array, Array, Array]:
        """Perform one training update with prioritized replay.

        Returns:
            Updated state, loss, mean Q-value, and mean TD-error.
        """
        key, sample_key = jax.random.split(key)
        batch = self.buffer.sample(state.buffer_state, sample_key)

        key, memory_key, next_memory_key = jax.random.split(key, 3)

        experience = batch.experience
        initial_carry = None
        initial_target_carry = None

        # Burn-in phase: warm up recurrent hidden state
        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], experience
            )
            initial_carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.params),
                burn_in.obs,
                burn_in.prev_done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.prev_done,
            )
            initial_carry = jax.lax.stop_gradient(initial_carry)
            initial_target_carry, (_, _) = self.q_network.apply(
                jax.lax.stop_gradient(state.target_params),
                burn_in.next_obs,
                burn_in.done,
                burn_in.action,
                add_feature_axis(burn_in.reward),
                burn_in.done,
            )
            initial_target_carry = jax.lax.stop_gradient(initial_target_carry)
            experience = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], experience
            )

        # Get target Q-values for n-step returns
        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            experience.next_obs,
            experience.done,
            experience.action,
            add_feature_axis(experience.reward),
            experience.done,
            initial_target_carry,
            rngs={"memory": next_memory_key},
        )

        # Double Q-learning: use online network to select action, target network to evaluate
        if self.cfg.double:
            _, (online_next_q_values, _) = self.q_network.apply(
                state.params,
                experience.next_obs,
                experience.done,
                experience.action,
                add_feature_axis(experience.reward),
                experience.done,
                initial_carry,
                rngs={"memory": memory_key},
            )
            next_actions = jnp.argmax(online_next_q_values, axis=-1)
            next_target_q_value = jnp.take_along_axis(
                next_target_q_values, add_feature_axis(next_actions), axis=-1
            )
            next_target_q_value = remove_feature_axis(next_target_q_value)
        else:
            next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        # Compute n-step returns
        learning_seq_len = experience.reward.shape[1]
        if self.cfg.n_step > 1 and learning_seq_len >= self.cfg.n_step:
            # Compute n-step targets
            n_step_targets = compute_n_step_returns(
                experience.reward,
                experience.done,
                next_target_q_value,
                self.cfg.n_step,
                self.cfg.gamma,
            )
            # Truncate experience to match n-step targets
            num_targets = n_step_targets.shape[1]
            experience = jax.tree.map(lambda x: x[:, :num_targets], experience)
            td_target = n_step_targets
        else:
            # Fall back to 1-step TD
            td_target = (
                experience.reward
                + (1 - experience.done) * self.cfg.gamma * next_target_q_value
            )

        # Compute importance sampling weights
        beta = self.beta_schedule(state.step)
        # Estimate buffer size from state
        buffer_size = jnp.where(
            state.buffer_state.is_full,
            self.cfg.buffer_size,
            state.buffer_state.current_index * self.cfg.num_envs,
        )
        buffer_size = jnp.maximum(buffer_size, 1)
        importance_weights = compute_importance_weights(
            batch.probabilities, buffer_size, beta
        )
        # Expand weights to match sequence dimension
        importance_weights = importance_weights[:, None]

        def loss_fn(params):
            hidden_state, (q_values, aux) = self.q_network.apply(
                params,
                experience.obs,
                experience.prev_done,
                experience.action,
                add_feature_axis(experience.reward),
                experience.prev_done,
                initial_carry,
                rngs={"memory": memory_key},
            )
            action = add_feature_axis(experience.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target

            # Importance-weighted MSE loss
            loss = (importance_weights * jnp.square(td_error)).mean()
            return loss, (q_value, td_error, hidden_state)

        (loss, (q_value, td_error, hidden_state)), grads = jax.value_and_grad(
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

        # Update priorities based on TD-error
        # Use mean absolute TD-error across sequence for each sample
        mean_td_error = jnp.abs(td_error).mean(axis=1)
        # Add small constant for stability
        new_priorities = mean_td_error + 1e-6
        buffer_state = self.buffer.set_priorities(
            state.buffer_state, batch.indices, new_priorities
        )

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return state, loss, q_value.mean(), mean_td_error.mean()

    def _learn(self, carry, _):
        """Execute training frequency steps and one update."""
        (key, state), transitions = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            carry,
            length=self.cfg.train_frequency // self.cfg.num_envs,
        )

        key, update_key = jax.random.split(key)
        state, loss, q_value, td_error = self._update(update_key, state)

        transitions.info["losses/loss"] = loss
        transitions.info["losses/q_value"] = q_value
        transitions.info["losses/td_error"] = td_error

        return (key, state), transitions.replace(obs=None, next_obs=None)

    @partial(jax.jit, static_argnames=["self"])
    def init(self, key):
        """Initialize algorithm state."""
        key, env_key, q_key, memory_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_envs,), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones(self.cfg.num_envs, dtype=jnp.bool)
        *_, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            env_keys, env_state, action, self.env_params
        )
        carry = self.q_network.initialize_carry(obs.shape)

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            carry,
        )
        target_params = self.q_network.init(
            {"params": q_key, "memory": memory_key},
            timestep.obs,
            timestep.done,
            timestep.action,
            add_feature_axis(timestep.reward),
            timestep.done,
            carry,
        )
        optimizer_state = self.optimizer.init(params)

        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=obs,
            done=done,
            info=info,
            prev_done=done,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return (
            key,
            R2D2State(
                step=0,
                timestep=timestep.from_sequence(),
                hidden_state=carry,
                env_state=env_state,
                params=params,
                target_params=target_params,
                optimizer_state=optimizer_state,
                buffer_state=buffer_state,
            ),
        )

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def warmup(self, key: Key, state: R2D2State, num_steps: int) -> tuple[Key, R2D2State]:
        """Fill replay buffer with random actions before learning."""
        (key, state), _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            (key, state),
            length=num_steps // self.cfg.num_envs,
        )
        return key, state

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def train(
        self,
        key: Key,
        state: R2D2State,
        num_steps: int,
    ):
        """Train for specified number of steps."""
        (
            (
                key,
                state,
            ),
            transitions,
        ) = jax.lax.scan(
            self._learn,
            (key, state),
            length=(num_steps // self.cfg.train_frequency),
        )
        return key, state, transitions

    @partial(jax.jit, static_argnames=["self", "num_steps"])
    def evaluate(self, key: Key, state: R2D2State, num_steps: int) -> tuple[Key, dict]:
        """Evaluate agent with greedy policy."""
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_eval_envs,), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_eval_envs,), dtype=jnp.float32)
        done = jnp.zeros(self.cfg.num_eval_envs, dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        hidden_state = self.q_network.initialize_carry(obs.shape)

        state = state.replace(
            timestep=timestep, hidden_state=hidden_state, env_state=env_state
        )

        (key, _), transitions = jax.lax.scan(
            partial(self._step, policy=self._greedy_action, write_to_buffer=False),
            (key, state),
            length=num_steps,
        )

        return key, transitions.replace(obs=None, next_obs=None)
