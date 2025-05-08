import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import chex
import flashbax as fbx
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro
from flax import core, struct
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper

import wandb
from networks import MaskedLSTMCell, MaskedOptimizedLSTMCell, MaskedRNN
from utils import LogWrapper

# from popjaxrl.envs import make
# from popjaxrl.envs.wrappers import AliasPrevActionV2


@dataclass(frozen=True)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_seeds: int = 1
    """the number of increasing seeds"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "memory_rl"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    debug: bool = False
    """whether to print debug information"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    sample_sequence_length: int | None = None
    """the sequence length of the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    @property
    def num_updates(self):
        return self.total_timesteps // self.num_envs


class QNetwork(nn.Module):
    action_dim: int
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray, initial_carry: jax.Array | None = None
    ):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        h, x = MaskedRNN(  # type: ignore
            self.cell,
            return_carry=True,
        )(x, mask, initial_carry=initial_carry)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return h, x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    done: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    next_done: chex.Array


class TrainState(TrainState):
    target_params: core.FrozenDict
    hidden_state: tuple
    initialize_carry: Callable = struct.field(pytree_node=False)
    timesteps: int
    n_updates: int


def make_env(args):
    env, env_params = gymnax.make(args.env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    return env, env_params


def make_train(args):

    env, env_params = make_env(args)

    q_network = QNetwork(
        action_dim=env.action_space(env_params).n,
        cell=MaskedOptimizedLSTMCell(
            128,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0),
        ),
    )

    sample_sequence_length: int = (  # type: ignore
        args.sample_sequence_length or env_params.max_steps_in_episode
    )
    assert (
        sample_sequence_length <= env_params.max_steps_in_episode
    ), "sample_sequence_length must be less than or equal to max_steps_in_episode"
    buffer = fbx.make_trajectory_buffer(
        add_batch_size=args.num_envs,
        sample_batch_size=args.num_envs,
        sample_sequence_length=sample_sequence_length,
        period=sample_sequence_length,
        min_length_time_axis=sample_sequence_length,
        max_length_time_axis=args.buffer_size,
    )

    def make_train_state(key):
        key, reset_key, q_key = jax.random.split(key, 3)

        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)
        initial_hidden_state = q_network.cell.initialize_carry(
            key, (args.num_envs, 128)
        )

        q_state = TrainState.create(
            apply_fn=q_network.apply,
            params=q_network.init(
                q_key,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                initial_hidden_state,
            ),
            target_params=q_network.init(
                q_key,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                initial_hidden_state,
            ),
            hidden_state=initial_hidden_state,
            tx=optax.adam(
                learning_rate=args.learning_rate,
            ),
            initialize_carry=q_network.cell.initialize_carry,
            timesteps=0,
            n_updates=0,
        )

        # Initialize buffer

        action = env.action_space().sample(key)
        _, state = env.reset(key, env_params)
        obs, _, reward, done, _ = env.step(key, state, action, env_params)
        transition = Transition(obs=obs, done=done, action=action, reward=reward, next_obs=obs, next_done=done)  # type: ignore
        buffer_state = buffer.init(transition)

        return key, q_state, buffer_state

    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return jnp.maximum(slope * t + start_e, end_e)

    @partial(jax.jit, static_argnums=(3))
    def train(key, q_state, buffer_state, num_steps):

        key, reset_key = jax.random.split(key)

        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)
        q_state = q_state.replace(
            hidden_state=q_network.cell.initialize_carry(key, (args.num_envs, 128))
        )

        def update_step(carry, _):

            key, q_state, buffer_state, state, obs, done = carry

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                q_state.timesteps,
            )

            key, step_key, action_key, sample_key = jax.random.split(key, 4)

            sample_key = jax.random.split(sample_key, args.num_envs)
            random_action = jax.vmap(env.action_space(env_params).sample)(sample_key)

            hidden_state, q_values = q_network.apply(
                q_state.params,
                jnp.expand_dims(obs, 1),
                jnp.expand_dims(done, 1),
                q_state.hidden_state,
            )
            q_state = q_state.replace(hidden_state=hidden_state)
            greedy_action = q_values.squeeze(1).argmax(axis=-1)

            action = jnp.where(
                jax.random.uniform(action_key, greedy_action.shape) < epsilon,
                random_action,
                greedy_action,
            )

            step_key = jax.random.split(step_key, args.num_envs)
            next_obs, state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, state, action, env_params)
            q_state = q_state.replace(timesteps=q_state.timesteps + args.num_envs)

            transition = Transition(obs=obs, done=done, action=action, reward=reward, next_obs=next_obs, next_done=next_done)  # type: ignore
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), transition)

            buffer_state = buffer.add(buffer_state, transition)

            def update(key, q_state):

                batch = buffer.sample(buffer_state, key).experience

                _, q_next_target = q_state.apply_fn(
                    q_state.target_params, batch.next_obs, batch.next_done
                )
                q_next_target = jnp.max(q_next_target, axis=-1)
                next_q_value = (
                    batch.reward + (1 - batch.next_done) * args.gamma * q_next_target
                )

                def loss_fn(params):
                    _, q_value = q_state.apply_fn(params, batch.obs, batch.done)
                    action = jnp.expand_dims(batch.action, axis=-1)
                    q_value = jnp.take_along_axis(q_value, action, axis=-1).squeeze(-1)
                    return jnp.square(q_value - next_q_value).mean()

                loss, grads = jax.value_and_grad(loss_fn)(q_state.params)
                q_state = q_state.apply_gradients(grads=grads)
                q_state = q_state.replace(n_updates=q_state.n_updates + 1)
                return q_state, loss

            key, update_key = jax.random.split(key)
            q_state, loss = jax.lax.cond(
                (
                    buffer.can_sample(buffer_state)
                    & (q_state.timesteps > args.learning_starts)
                    & (q_state.timesteps % args.train_frequency == 0)
                ),
                lambda key, q_state: update(key, q_state),
                lambda _, q_state: (q_state, jnp.array(0.0)),
                update_key,
                q_state,
            )

            target_params = optax.incremental_update(
                q_state.params,
                q_state.target_params,
                args.tau,
            )
            q_state = jax.lax.cond(
                q_state.timesteps % args.target_network_frequency == 0,
                lambda q_state: q_state.replace(
                    target_params=target_params,
                ),
                lambda train_state: train_state,
                operand=q_state,
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

                jax.debug.callback(callback, info)

            return (
                key,
                q_state,
                buffer_state,
                state,
                next_obs,
                next_done,
            ), info

        (key, q_state, buffer_state, *_), info = jax.lax.scan(
            update_step,
            (key, q_state, buffer_state, state, obs, done),
            length=num_steps,
        )
        return key, q_state, buffer_state, info

    return train, make_train_state


def main():
    args = tyro.cli(Args)

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)

    train, make_train_state = make_train(args)

    key = jax.random.key(args.seed)

    key, q_state, buffer_state = make_train_state(key)
    key, q_state, buffer_state, info = train(
        key, q_state, buffer_state, args.num_updates
    )

    import matplotlib.pyplot as plt

    plt.plot(info["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    main()
