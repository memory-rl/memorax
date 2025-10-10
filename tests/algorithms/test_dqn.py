import jax

from memory_rl.algorithms import DQN, DQNConfig


def test_dqn_init_and_train(
    rng_key: jax.Array,
    dummy_env,
    dummy_env_params,
    dqn_components,
):
    buffer, q_network, optimizer, schedule = dqn_components

    cfg = DQNConfig(
        name="dqn-smoke",
        learning_rate=3e-4,
        num_envs=2,
        num_eval_envs=1,
        buffer_size=16,
        gamma=0.99,
        tau=0.005,
        target_network_frequency=4,
        batch_size=4,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.2,
        learning_starts=0,
        train_frequency=4,
        buffer=buffer,
        feature_extractor=q_network.feature_extractor,
        torso=q_network.torso,
        double=False,
    )

    algorithm = DQN(
        cfg=cfg,
        env=dummy_env,
        env_params=dummy_env_params,
        q_network=q_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=schedule,
    )

    key, state = algorithm.init(rng_key)

    key, state = algorithm.warmup(key, state, num_steps=cfg.train_frequency)

    key, state, transitions = algorithm.train(key, state, num_steps=cfg.train_frequency)

    eval_key, _ = algorithm.evaluate(key, state, num_steps=2)
