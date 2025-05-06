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

from utils import LogWrapper, RecordEpisodeStatistics, compute_gae

# from popjaxrl.envs import make
# from popjaxrl.envs.wrappers import AliasPrevActionV2


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

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
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
        return logits.squeeze(), value.squeeze()


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
    env = LogWrapper(env)
    env = FlattenObservationWrapper(env)

    key = jax.random.key(args.seed)
    key, reset_key, agent_key = jax.random.split(key, 3)

    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    lr_schedule = linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=(
            args.num_iterations * args.update_epochs * args.num_minibatches
        ),
    )
    learning_rate = lr_schedule if args.anneal_lr else args.learning_rate

    agent = Agent(action_dim=env.action_space(env_params).n)
    agent_state = TrainState.create(
        apply_fn=agent.apply,
        params=agent.init(agent_key, obs),
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

        def update_step(carry: tuple, _):

            def step(carry: tuple, _):
                key, obs, state, agent_state = carry

                key, action_key, step_key = jax.random.split(key, 3)

                logits, value = agent_state.apply_fn(agent_state.params, obs)
                probs = distrax.Categorical(logits=logits)
                action = probs.sample(seed=action_key)
                log_prob = probs.log_prob(action)

                step_key = jax.random.split(step_key, args.num_envs)
                next_obs, state, reward, done, info = jax.vmap(
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
                    agent_state,
                )
                return carry, transition

            (key, final_obs, state, agent_state), transitions = jax.lax.scan(
                step,
                carry,
                length=args.num_steps,
            )
            _, final_value = agent_state.apply_fn(agent_state.params, final_obs)

            advantages, returns = compute_gae(
                args.gamma, args.gae_lambda, final_value, transitions
            )

            if args.norm_adv:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            batch = (transitions, advantages, returns)

            def update_epoch(carry: tuple, _):
                key, agent_state, batch = carry

                def update_minibatch(agent_state: TrainState, minibatch: tuple):
                    transitions, advantages, returns = minibatch

                    def loss_fn(params, transitions, advantages, returns):
                        logits, value = agent_state.apply_fn(
                            params, transitions.observation
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
                        agent_state.params, transitions, advantages, returns
                    )
                    agent_state = agent_state.apply_gradients(grads=agent_grads)
                    return agent_state, loss

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(permutation_key, args.batch_size)
                flattened_batch = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), flattened_batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [args.num_minibatches, -1] + list(x.shape[1:])
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
                    batch,
                ), loss

            (key, agent_state, batch), _ = jax.lax.scan(
                update_epoch,
                (key, agent_state, batch),
                length=args.update_epochs,
            )
            transitions, *_ = batch

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

            return (key, final_obs, state, agent_state), transitions.info

        (key, _, _, agent_state), info = jax.lax.scan(
            update_step,
            (key, obs, state, agent_state),
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

        def step(carry, _):
            key, obs, state = carry

            logits, _ = agent_state.apply_fn(agent_state.params, obs)
            action = jnp.argmax(logits, axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, 4)
            obs, state, _, _, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_key, state, action, env_params
            )

            return (key, obs, state), info

        (key, obs, state), info = jax.lax.scan(
            step, (key, obs, state), length=num_steps
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
