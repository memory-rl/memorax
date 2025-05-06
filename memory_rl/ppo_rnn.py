import os
from dataclasses import dataclass
from functools import partial

import chex
import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro
from flax.training.train_state import TrainState
from gymnax.wrappers import FlattenObservationWrapper
from optax import linear_schedule

from networks import MaskedGRUCell, MaskedRNN
from utils import LogWrapper
from utils import compute_recurrent_gae as compute_gae


@dataclass(frozen=True)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "jax_ppo"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    debug: bool = False
    """if toggled, this experiment will run in debug mode"""

    # Algorithm specific arguments
    env_id: str = "Breakout-MinAtar"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
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

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self):
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self):
        return self.total_timesteps // self.batch_size


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
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        h, x = MaskedRNN(
            self.cell,
            time_major=True,
            return_carry=True,
        )(x, mask, initial_carry=initial_carry)
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return h, logits.squeeze(), value.squeeze()


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array


def make_train(args: Args):

    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    key = jax.random.key(args.seed)
    key, reset_key, agent_key = jax.random.split(key, 3)

    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)

    lr_schedule = linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=(
            args.num_iterations * args.update_epochs * args.num_minibatches
        ),
    )

    learning_rate = lr_schedule if args.anneal_lr else args.learning_rate

    agent = Agent(action_dim=env.action_space(env_params).n, cell=MaskedGRUCell(128))
    agent_state = TrainState.create(
        apply_fn=agent.apply,
        params=agent.init(agent_key, jnp.expand_dims(obs, 0), jnp.expand_dims(done, 0)),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        ),
    )

    @partial(jax.jit, static_argnums=(2,))
    def train(key, agent_state, num_steps):
        key, reset_key = jax.random.split(key)

        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)
        initial_hidden = agent.cell.initialize_carry(key, obs.shape)

        def update_step(carry: tuple, _):

            def step(carry: tuple, _):
                key, obs, state, done, agent_state, hidden = carry

                key, action_key, step_key = jax.random.split(key, 3)

                hidden, logits, value = agent_state.apply_fn(
                    agent_state.params,
                    jnp.expand_dims(obs, 0),
                    jnp.expand_dims(done, 0),
                    hidden,
                )
                probs = distrax.Categorical(logits=logits)
                action = probs.sample(seed=action_key)
                log_prob = probs.log_prob(action)

                step_key = jax.random.split(step_key, args.num_envs)
                next_obs, state, reward, next_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_key, state, action, env_params)
                transition = Transition(
                    observation=obs,  # type: ignore
                    action=action,  # type: ignore
                    reward=reward,  # type: ignore
                    done=done,  # type: ignore
                    info=info,  # type: ignore
                    log_prob=log_prob,  # type: ignore
                    value=value,  # type: ignore
                )

                carry = (
                    key,
                    next_obs,
                    state,
                    next_done,
                    agent_state,
                    hidden,
                )
                return carry, transition

            *_, initial_hidden = carry
            (key, final_obs, state, final_done, agent_state, hidden), transitions = (
                jax.lax.scan(
                    step,
                    carry,
                    length=args.num_steps,
                )
            )
            _, _, final_value = agent_state.apply_fn(
                agent_state.params,
                jnp.expand_dims(final_obs, 0),
                jnp.expand_dims(final_done, 0),
                hidden,
            )

            advantages, returns = compute_gae(
                args.gamma, args.gae_lambda, transitions, final_value, final_done
            )

            if args.norm_adv:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            def update_epoch(carry: tuple, _):
                key, agent_state, initial_hidden, transitions, advantages, returns = (
                    carry
                )

                def update_minibatch(agent_state: TrainState, minibatch: tuple):
                    initial_hidden, transitions, advantages, returns = minibatch

                    def loss_fn(
                        params, initial_hidden, transitions, advantages, returns
                    ):
                        _, logits, value = agent_state.apply_fn(
                            params,
                            transitions.observation,
                            transitions.done,
                            initial_hidden.squeeze(0),
                        )

                        probs = distrax.Categorical(logits=logits)
                        log_prob = probs.log_prob(transitions.action)
                        entropy = probs.entropy().mean()

                        ratio = jnp.exp(log_prob - transitions.log_prob)
                        actor_loss = -jnp.minimum(
                            ratio * advantages,
                            jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                            * advantages,
                        ).mean()

                        if args.clip_vloss:
                            critic_loss = jnp.square(value - returns)
                            clipped_value = transitions.value + jnp.clip(
                                (value - transitions.value),
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            clipped_critic_loss = jnp.square(clipped_value - returns)
                            critic_loss = (
                                1
                                / 2
                                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
                            )
                        else:
                            critic_loss = 1 / 2 * jnp.square(value - returns).mean()

                        return (
                            actor_loss
                            + args.vf_coef * critic_loss
                            - args.ent_coef * entropy
                        )

                    grad_fn = jax.value_and_grad(loss_fn)
                    loss, agent_grads = grad_fn(
                        agent_state.params,
                        initial_hidden,
                        transitions,
                        advantages,
                        returns,
                    )
                    agent_state = agent_state.apply_gradients(grads=agent_grads)
                    return agent_state, loss

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(permutation_key, args.num_envs)

                batch = (initial_hidden, transitions, advantages, returns)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], args.num_minibatches, -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                agent_state, loss = jax.lax.scan(
                    update_minibatch,
                    agent_state,
                    minibatches,
                )
                return (
                    key,
                    agent_state,
                    initial_hidden,
                    transitions,
                    advantages,
                    returns,
                ), loss

            (key, agent_state, *_), _ = jax.lax.scan(
                update_epoch,
                (
                    key,
                    agent_state,
                    jnp.expand_dims(initial_hidden, 0),
                    transitions,
                    advantages,
                    returns,
                ),
                length=args.update_epochs,
            )

            if args.debug:

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                        if args.track:
                            wandb.log(
                                {"episodic return": return_values[t]}, step=timesteps[t]
                            )

                jax.debug.callback(callback, transitions.info)

            return (
                key,
                final_obs,
                state,
                final_done,
                agent_state,
                hidden,
            ), transitions.info

        (key, _, _, _, agent_state, _), info = jax.lax.scan(
            update_step,
            (key, obs, state, done, agent_state, initial_hidden),
            length=num_steps,
        )

        return key, agent_state, info

    return train, agent_state


def make_evaluate(args: Args):
    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    @partial(jax.jit, static_argnums=(2))
    def evaluate(key, agent_state, num_steps):

        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, 4)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)
        # initial_hidden = agent.cell.initialize_carry(key, obs.shape)
        initial_hidden = MaskedGRUCell(128).initialize_carry(key, obs.shape)

        def step(carry, _):
            key, obs, state, done, hidden = carry

            hidden, logits, _ = agent_state.apply_fn(
                agent_state.params,
                jnp.expand_dims(obs, 0),
                jnp.expand_dims(done, 0),
                hidden,
            )
            action = jnp.argmax(logits, axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, 4)
            obs, state, _, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, state, action, env_params
            )

            return (key, obs, state, done, hidden), info

        (key, *_), info = jax.lax.scan(
            step, (key, obs, state, done, initial_hidden), length=num_steps
        )

        return key, info

    return evaluate


if __name__ == "__main__":
    args = tyro.cli(Args)

    import wandb

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.exp_name,
        )

    train, agent_state = make_train(args)
    evaluate = make_evaluate(args)

    key = jax.random.key(args.seed)
    key, agent_state, info = train(key, agent_state, args.num_iterations)
    key, info = evaluate(key, agent_state, 1000)

    returns = info["returned_episode_returns"][info["returned_episode"]].mean()

    print("Mean Episodic Return: ", returns)
