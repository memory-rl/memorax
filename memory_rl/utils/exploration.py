import jax
import jax.numpy as jnp


def epsilon_greedy(key, env, env_params, num_envs, q_network, q_state, epsilon, obs):
    key, action_key, sample_key = jax.random.split(key, 3)

    sample_key = jax.random.split(sample_key, num_envs)
    random_action = jax.vmap(lambda key: env.action_space(env_params).sample(key))(
        sample_key
    )

    q_values = q_network.apply(q_state.params, obs)
    greedy_action = q_values.argmax(axis=-1)

    action = jnp.where(
        jax.random.uniform(action_key, greedy_action.shape) < epsilon,
        random_action,
        greedy_action,
    )
    return action


def recurrent_epsilon_greedy(
    key, env, env_params, num_envs, q_network, q_state, epsilon, obs, done
):
    key, action_key, sample_key = jax.random.split(key, 3)

    sample_key = jax.random.split(sample_key, num_envs)
    random_action = jax.vmap(lambda key: env.action_space(env_params).sample(key))(
        sample_key
    )

    q_values = q_network.apply(q_state.params, obs, done)
    greedy_action = q_values.argmax(axis=-1)

    action = jnp.where(
        jax.random.uniform(action_key, greedy_action.shape) < epsilon,
        random_action,
        greedy_action,
    )
    return action
