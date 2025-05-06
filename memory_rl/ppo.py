import os
from dataclasses import dataclass

import jax

import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from optax import linear_schedule
import chex
import distrax
import gymnax
from popjaxrl.envs import make
from popjaxrl.envs.wrappers import AliasPrevActionV2

import tyro

from utils import compute_gae
from utils import LogWrapper


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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
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
    target_kl: float | None = None
    """the target KL divergence threshold"""

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self):
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self):
        return self.total_timesteps // self.batch_size


class Critic(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return value.squeeze()


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            64,
            kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return logits.squeeze()


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array


def make_train(key, args, env, env_params, actor, critic):

    key, reset_key, actor_key, critic_key = jax.random.split(key, 4)

    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    lr_schedule = linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=(
            args.num_iterations * args.update_epochs * args.num_minibatches
        ),
    )

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        ),
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, obs),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        ),
    )

    def update_step(carry: tuple, _):

        def step(carry: tuple, _):
            key, obs, state, actor_state, critic_state = carry

            key, action_key, step_key = jax.random.split(key, 3)

            logits = actor_state.apply_fn(actor_state.params, obs)
            probs = distrax.Categorical(logits=logits)
            action = probs.sample(seed=action_key)
            log_prob = probs.log_prob(action)
            value = critic_state.apply_fn(critic_state.params, obs)

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
                actor_state,
                critic_state,
            )
            return carry, transition

        (key, final_obs, state, actor_state, critic_state), transitions = jax.lax.scan(
            step,
            carry,
            length=args.num_steps,
        )
        final_value = critic_state.apply_fn(critic_state.params, final_obs)

        advantages, returns = compute_gae(
            args.gamma, args.gae_lambda, final_value, transitions
        )

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = (transitions, advantages, returns)

        def update_epoch(carry: tuple, _):
            key, actor_state, critic_state, batch = carry

            def update_minibatch(carry: tuple, minibatch: tuple):
                transitions, advantages, returns = minibatch

                def actor_loss(params, transitions, advantages):
                    logits = actor_state.apply_fn(params, transitions.observation)
                    probs = distrax.Categorical(logits=logits)
                    log_prob = probs.log_prob(transitions.action)
                    entropy = probs.entropy()

                    ratio = jnp.exp(log_prob - transitions.log_prob)
                    loss = -jnp.minimum(
                        ratio * advantages,
                        jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                        * advantages,
                    ).mean()
                    return loss - args.ent_coef * entropy.mean()

                def critic_loss(params, transitions, returns):
                    value = critic_state.apply_fn(params, transitions.observation)
                    if args.clip_vloss:
                        loss = jnp.square(value - returns)
                        clipped_value = transitions.value + jnp.clip(
                            (value - transitions.value), -args.clip_coef, args.clip_coef
                        )
                        clipped_loss = jnp.square(clipped_value - returns)
                        loss = 1 / 2 * jnp.maximum(loss, clipped_loss).mean()
                    else:
                        loss = 1 / 2 * jnp.square(value - returns).mean()
                    return args.vf_coef * loss

                actor_state, critic_state = carry

                actor_grad_fn = jax.value_and_grad(actor_loss)
                actor_loss, actor_grads = actor_grad_fn(
                    actor_state.params, transitions, advantages
                )
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                critic_grad_fn = jax.value_and_grad(critic_loss)
                critic_loss, critic_grads = critic_grad_fn(
                    critic_state.params, transitions, returns
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                carry = (actor_state, critic_state)
                loss = (actor_loss, critic_loss)
                return carry, loss

            key, permutation_key = jax.random.split(key)

            permutation = jax.random.permutation(permutation_key, args.batch_size)
            flattened_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), flattened_batch
            )
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [args.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (actor_state, critic_state), loss = jax.lax.scan(
                update_minibatch,
                (actor_state, critic_state),
                minibatches,
            )
            return (
                key,
                actor_state,
                critic_state,
                batch,
            ), loss

        (key, actor_state, critic_state, batch), _ = jax.lax.scan(
            update_epoch,
            (key, actor_state, critic_state, batch),
            length=args.update_epochs,
        )
        transitions, *_ = batch

        if args.debug:

            def callback(info):
                return_values = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                timesteps = info["timestep"][info["returned_episode"]] * args.num_envs
                for t in range(len(timesteps)):
                    print(
                        f"global step={timesteps[t]}, episodic return={return_values[t]}"
                    )
                    wandb.log({"episodic return": return_values[t]}, step=timesteps[t])

            jax.debug.callback(callback, transitions.info)

        return (key, final_obs, state, actor_state, critic_state), transitions.info

    (key, obs, state, actor_state, critic_state), info = jax.lax.scan(
        update_step,
        (key, obs, state, actor_state, critic_state),
        length=args.num_iterations,
    )

    return key, actor_state, critic_state, info


if __name__ == "__main__":
    make_train = jax.jit(make_train, static_argnums=(1, 2, 3, 4, 5))
    args = tyro.cli(Args)

    import wandb

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.exp_name,
        )

    # env, env_params = gymnax.make(args.env_id)
    # env = LogWrapper(env)
    env, env_params = make(args.env_id)
    env = AliasPrevActionV2(env)
    env = LogWrapper(env)

    key = jax.random.key(args.seed)

    actor = Actor(action_dim=env.action_space(env_params).n)
    critic = Critic()

    key, actor_state, critic_state, info = make_train(
        key, args, env, env_params, actor, critic
    )
