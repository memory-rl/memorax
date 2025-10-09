import jax

from memory_rl.algorithms import PQN, PQNConfig


def test_pqn_init_and_train(
    rng_key: jax.Array,
    dummy_env,
    dummy_env_params,
    pqn_components,
):
    q_network, optimizer, epsilon_schedule = pqn_components

    cfg = PQNConfig(
        name="pqn-smoke",
        learning_rate=3e-4,
        num_envs=2,
        num_eval_envs=1,
        num_steps=2,
        gamma=0.99,
        td_lambda=0.9,
        num_minibatches=1,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.1,
        update_epochs=1,
        max_grad_norm=0.5,
        learning_starts=0,
        actor=None,
        critic=None,
    )

    algorithm = PQN(
        cfg=cfg,
        env=dummy_env,
        env_params=dummy_env_params,
        q_network=q_network,
        optimizer=optimizer,
        epsilon_schedule=epsilon_schedule,
    )

    key, state = algorithm.init(rng_key)
    key, state = algorithm.warmup(key, state, num_steps=cfg.batch_size)
    rollout_steps = cfg.num_envs * cfg.num_steps
    key, state, transitions = algorithm.train(key, state, num_steps=rollout_steps)
    eval_key, eval_transitions = algorithm.evaluate(key, state, num_steps=2)
