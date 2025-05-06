import os
import jax
import jax.numpy as jnp
from typing import Any
from dataclasses import dataclass

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper
import gymnax
import flashbax as fbx
import tyro
from popjaxrl.envs import make
from popjaxrl.envs.wrappers import AliasPrevActionV2

from wrapper import LogWrapper
from networks import MaskedRNN, MaskedLSTMCell
from exploration import recurrent_epsilon_greedy as epsilon_greedy


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
    wandb_project_name: str = "jax"
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

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = MaskedRNN(
            MaskedLSTMCell(128),
        )(x, mask)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


def make_train(args):

    env, env_params = gymnax.make(args.env_id)
    # env, env_params = make(args.env_id)
    # env = AliasPrevActionV2(env)
    env = LogWrapper(env)

    key = jax.random.key(args.seed)

    # buffer = fbx.make_flat_buffer(
    #     max_length=args.buffer_size,
    #     min_length=args.batch_size,
    #     sample_batch_size=args.batch_size,
    #     add_sequences=False,
    #     add_batch_size=args.num_envs,
    # )
    buffer = fbx.make_trajectory_buffer(
        add_batch_size=args.num_envs,
        sample_batch_size=args.num_envs,
        sample_sequence_length=args.batch_size,
        period=args.batch_size - 1,
        min_length_time_axis=0,
        max_size=args.buffer_size,
    )

    action = env.action_space().sample(key)
    _, state = env.reset(key, env_params)
    obs, _, reward, done, _ = env.step(key, state, action, env_params)
    transition = Transition(obs=obs, action=action, reward=reward, done=done)  # type: ignore
    buffer_state = buffer.init(transition)

    key, reset_key, q_key = jax.random.split(key, 3)
    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)

    q_network = QNetwork(action_dim=env.action_space(env_params).n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs, done),
        target_params=q_network.init(q_key, obs, done),
        tx=optax.adam(
            learning_rate=args.learning_rate,
        ),
        timesteps=0,
        n_updates=0,
    )

    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return jnp.maximum(slope * t + start_e, end_e)

    def train(key, q_state, buffer_state):

        key, reset_key = jax.random.split(key)

        reset_key = jax.random.split(reset_key, args.num_envs)
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        done = jax.vmap(env.is_terminal, in_axes=(0, None))(state.env_state, env_params)

        def update_step(carry, _):

            key, q_state, buffer_state, state, obs, done = carry

            key, exploration_key, step_key = jax.random.split(key, 3)
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                q_state.timesteps,
            )
            action = epsilon_greedy(
                exploration_key,
                env,
                env_params,
                args.num_envs,
                q_network,
                q_state,
                epsilon,
                obs,
                done,
            )

            step_key = jax.random.split(step_key, args.num_envs)
            next_obs, state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, state, action, env_params)
            q_state = q_state.replace(timesteps=q_state.timesteps + args.num_envs)

            transition = Transition(obs=obs, action=action, reward=reward, done=done)  # type: ignore
            transition = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), transition)
            buffer_state = buffer.add(buffer_state, transition)
            exit()

            def update(key, q_state):

                batch = buffer.sample(buffer_state, key).experience

                q_next_target = q_network.apply(
                    q_state.target_params, batch.second.obs, batch.second.done
                )
                q_next_target = jnp.max(q_next_target, axis=-1)
                next_q_value = (
                    batch.first.reward
                    + (1 - batch.first.done) * args.gamma * q_next_target
                )

                def loss_fn(params):
                    q_value = q_network.apply(params, batch.first.obs, batch.first.done)
                    action = jnp.expand_dims(batch.first.action, axis=-1)
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

            metrics = {
                "timesteps": q_state.timesteps,
                "updates": q_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

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
                        wandb.log(
                            {"episodic return": return_values[t]}, step=timesteps[t]
                        )

                jax.debug.callback(callback, info)

            return (key, q_state, buffer_state, state, next_obs, done), metrics

        runner_state, metrics = jax.lax.scan(
            update_step,
            (key, q_state, buffer_state, state, obs, done),
            length=args.num_updates,
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, q_state, buffer_state


def main():
    args = tyro.cli(Args)

    if args.debug:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)

    train, q_state, buffer_state = make_train(args)
    train = jax.jit(train)

    key = jax.random.key(args.seed)
    train(key, q_state, buffer_state)


if __name__ == "__main__":
    main()
