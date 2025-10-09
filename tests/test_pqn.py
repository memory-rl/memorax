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
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.9,
        num_minibatches=1,
        update_epochs=1,
        normalize_advantage=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.0,
        vf_coef=0.5,
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
    train_steps = cfg.batch_size
    key, state, transitions = algorithm.train(key, state, num_steps=train_steps)
    eval_key, eval_transitions = algorithm.evaluate(key, state, num_steps=2)
