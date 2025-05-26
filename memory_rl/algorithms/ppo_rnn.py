import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro
from flax import core, struct
from flax.training.train_state import TrainState
from gymnax.wrappers import FlattenObservationWrapper
from optax import linear_schedule
from recurrent_networks import MaskedGRUCell, MaskedRNN

from utils import LogWrapper
from utils import compute_recurrent_gae as compute_gae


@dataclass(frozen=True)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_seeds: int = 1
    """the number of seeds to run"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "memory_rl"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    debug: bool = False
    """if toggled, this experiment will run in debug mode"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    learning_starts: int = 0
    """the number of steps before learning starts"""
    num_train_steps: int = 100_000
    """the number of steps per epoch"""
    num_evaluation_steps: int = 50_000

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def num_iterations(self):
        return self.total_timesteps // self.batch_size

    @property
    def num_epochs(self):
        return self.total_timesteps // self.num_train_steps + 1


class Agent(nn.Module):
    action_dim: int
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
    ):
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        h, x = MaskedRNN(  # type: ignore
            self.cell,
            time_major=False,
            return_carry=True,
        )(x, mask, initial_carry=initial_carry)
        x = nn.Dense(
            128,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        probs = distrax.Categorical(logits=logits)

        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return h, probs, value.squeeze(-1)

    def initialize_carry(self, input_shape):
        return self.cell.initialize_carry(jax.random.key(0), input_shape)


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class RPPOState:
    step: int
    obs: chex.Array
    done: chex.Array
    env_state: gymnax.EnvState
    hidden_state: chex.Array
    params: core.FrozenDict[str, chex.ArrayTree]
    optimizer_state: optax.OptState


@chex.dataclass(frozen=True)
class RPPO:
    args: Args
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    agent: Agent
    optimizer: optax.GradientTransformation

    def init(self, key):
        key, env_key, agent_key = jax.random.split(key, 3)

        env_keys = jax.random.split(env_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        done = jnp.zeros(self.args.num_envs, dtype=bool)
        hidden_state = self.agent.initialize_carry(jax.random.key(0), (self.args.num_envs, 128))

        dummy_x_for_init = jnp.expand_dims(obs, 1)
        dummy_mask_for_init = jnp.expand_dims(done, 1)
        params = self.agent.init(agent_key, dummy_x_for_init, dummy_mask_for_init, initial_carry=hidden_state)['params']

        optimizer_state = self.optimizer.init(params)

        return (
            key,
            RPPOState(
                step=0,
                obs=obs,
                done=done,
                hidden_state=hidden_state,
                env_state=env_state,
                params=params,
                optimizer_state=optimizer_state,
            ),
        )

    def warmup(self, key, state, num_steps):
        """No warmup needed for RPPO"""
        return key, state

    def train(self, key, state, num_steps):

        def update_step(carry: tuple, _):

            def step(carry: tuple, _):
                key, state = carry

                key, action_key, step_key = jax.random.split(key, 3)

                obs_for_agent = jnp.expand_dims(state.obs, 1)
                done_for_agent = jnp.expand_dims(state.done, 1)

                hidden_state, probs, value = self.agent.apply(
                    state.params,
                    obs_for_agent,
                    done_for_agent,
                    state.hidden_state,
                )
                value = value.squeeze(1)
                action = probs.sample(seed=action_key).squeeze(1)
                log_prob = probs.log_prob(action).squeeze(1)

                step_key = jax.random.split(step_key, self.args.num_envs)
                next_obs, env_state, reward, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(step_key, state.env_state, action, self.env_params)
                transition = Transition(
                    observation=state.obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                    log_prob=log_prob,
                    value=value,
                )

                state = state.replace(
                    step=state.step + self.args.num_envs,
                    obs=next_obs,
                    env_state=env_state,
                    done=done,
                    hidden_state=hidden_state,
                )
                return (key, state), transition

            key, state = carry
            initial_hidden_state = state.hidden_state
            (key, state), transitions = jax.lax.scan(
                step,
                (key, state),
                length=self.args.num_steps,
            )
            obs_for_final_value = jnp.expand_dims(state.obs, 1)
            done_for_final_value = jnp.expand_dims(state.done, 1)
            _, _, final_value = self.agent.apply(
                state.params,
                obs_for_final_value,
                done_for_final_value,
                state.hidden_state,
            )

            final_value = final_value.squeeze(1)

            advantages, returns = compute_gae(
                self.args.gamma,
                self.args.gae_lambda,
                transitions,
                final_value,
                state.done,
            )

            if self.args.norm_adv:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            def update_epoch(carry: tuple, _):

                def update_minibatch(state, minibatch: tuple):
                    initial_hidden, transitions, advantages, returns = minibatch

                    def loss_fn(
                        params, initial_hidden, transitions, advantages, returns
                    ):
                        obs_b_t_f = jnp.swapaxes(transitions.observation, 0, 1)
                        mask_b_t = jnp.swapaxes(transitions.done, 0, 1)

                        _, probs, value = self.agent.apply(
                            params,
                            obs_b_t_f,
                            mask_b_t,
                            initial_hidden,
                        )

                        actions_b_t = jnp.swapaxes(transitions.action, 0, 1)
                        log_prob_b_t_prev = jnp.swapaxes(transitions.log_prob, 0, 1)
                        advantages_b_t = jnp.swapaxes(advantages, 0, 1)
                        returns_b_t = jnp.swapaxes(returns, 0, 1)
                        value_b_t_prev = jnp.swapaxes(transitions.value, 0, 1)

                        log_prob = probs.log_prob(actions_b_t)
                        entropy = probs.entropy().mean()

                        ratio = jnp.exp(log_prob - log_prob_b_t_prev)
                        actor_loss = -jnp.minimum(
                            ratio * advantages_b_t,
                            jnp.clip(
                                ratio,
                                1.0 - self.args.clip_coef,
                                1.0 + self.args.clip_coef,
                            )
                            * advantages_b_t,
                        ).mean()

                        if self.args.clip_vloss:
                            critic_loss = jnp.square(value - returns_b_t)
                            clipped_value = value_b_t_prev + jnp.clip(
                                (value - value_b_t_prev),
                                -self.args.clip_coef,
                                self.args.clip_coef,
                            )
                            clipped_critic_loss = jnp.square(clipped_value - returns_b_t)
                            critic_loss = (
                                0.5
                                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
                            )
                        else:
                            critic_loss = 0.5 * jnp.square(value - returns_b_t).mean()

                        loss = (
                            actor_loss
                            + self.args.vf_coef * critic_loss
                            - self.args.ent_coef * entropy
                        )

                        return loss, (actor_loss, critic_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss, (actor_loss, critic_loss, entropy_loss)), grads = grad_fn(
                        state.params,
                        initial_hidden,
                        transitions,
                        advantages,
                        returns,
                    )
                    updates, optimizer_state = self.optimizer.update(
                        grads, state.optimizer_state, state.params
                    )
                    params = optax.apply_updates(state.params, updates)

                    state = state.replace(
                        params=params, optimizer_state=optimizer_state
                    )
                    return state, (loss, actor_loss, critic_loss, entropy_loss)

                key, state, initial_hidden, transitions, advantages, returns = carry

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(
                    permutation_key, self.args.num_envs
                )

                initial_hidden_shuffled = jnp.take(initial_hidden, permutation, axis=0)

                initial_hidden_minibatches = jnp.reshape(
                    initial_hidden_shuffled,
                    [self.args.num_minibatches, -1] + list(initial_hidden_shuffled.shape[1:])
                )

                transitions_shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), transitions)
                advantages_shuffled = jnp.take(advantages, permutation, axis=1)
                returns_shuffled = jnp.take(returns, permutation, axis=1)

                def reshape_and_swap(x):
                    return jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], self.args.num_minibatches, -1] + list(x.shape[2:])
                        ), 1, 0
                    )

                transitions_minibatches = jax.tree_util.tree_map(reshape_and_swap, transitions_shuffled)
                advantages_minibatches = reshape_and_swap(advantages_shuffled)
                returns_minibatches = reshape_and_swap(returns_shuffled)

                minibatches_for_scan = (
                    initial_hidden_minibatches, 
                    transitions_minibatches, 
                    advantages_minibatches, 
                    returns_minibatches
                )
                
                state, loss = jax.lax.scan(
                    update_minibatch,
                    state, 
                    minibatches_for_scan,
                )

                return (
                    key,
                    state,
                    initial_hidden_state,
                    transitions,
                    advantages,
                    returns,
                ), loss

            (key, state, *_), losses = jax.lax.scan(
                update_epoch,
                (
                    key,
                    state,
                    initial_hidden_state,
                    transitions,
                    advantages,
                    returns,
                ),
                length=self.args.update_epochs,
            )

            if self.args.debug:

                def callback(info, losses):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * self.args.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, transitions.info, losses)

            return (
                key,
                state,
            ), transitions.info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=num_steps // (self.args.num_envs * self.args.num_steps),
        )

        return key, state, info

    def evaluate(self, key, state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.args.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        done = jnp.zeros(self.args.num_envs, dtype=bool)
        initial_hidden_state = self.agent.initialize_carry(jax.random.key(0), (self.args.num_envs, 128))

        def step(carry, _):
            key, obs, done, env_state, hidden_state = carry

            obs_for_agent = jnp.expand_dims(obs, 1)
            done_for_agent = jnp.expand_dims(done, 1)

            hidden_state, probs, _ = self.agent.apply(
                state.params,
                obs_for_agent,
                done_for_agent,
                hidden_state,
            )
            action = jnp.argmax(probs.logits, axis=-1).squeeze(1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.args.num_envs)
            obs, env_state, _, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, done, env_state, hidden_state), info

        (key, *_), info = jax.lax.scan(
            step, (key, obs, done, env_state, initial_hidden_state), length=num_steps
        )

        return key, info


def make_rppo_old(args: Args):
    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key, 2)

    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    done = jnp.zeros(args.num_envs, dtype=bool)

    agent = Agent(
        action_dim=env.action_space(env_params).n,
        cell=MaskedGRUCell(
            128,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0),
        ),
    )

    if args.anneal_lr:
        learning_rate = linear_schedule(
            init_value=args.learning_rate,
            end_value=0.0,
            transition_steps=(
                args.num_iterations * args.update_epochs * args.num_minibatches
            ),
        )
    else:
        learning_rate = args.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    return RPPO(
        args=args,
        env=env,
        env_params=env_params,
        agent=agent,
        optimizer=optimizer,
    )
