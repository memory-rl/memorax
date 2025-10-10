import jax

from memory_rl.algorithms import SAC, SACConfig


def test_sac_init_and_train(
    rng_key: jax.Array,
    dummy_continuous_env,
    dummy_continuous_env_params,
    sac_components,
):
    (
        buffer,
        actor_network,
        critic_network,
        alpha_network,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
    ) = sac_components

    cfg = SACConfig(
        name="sac-smoke",
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=1e-3,
        num_envs=2,
        num_eval_envs=1,
        buffer_size=32,
        gamma=0.99,
        tau=0.01,
        train_frequency=4,
        target_update_frequency=4,
        batch_size=4,
        initial_alpha=0.1,
        target_entropy_scale=1.0,
        learning_starts=0,
        max_grad_norm=1.0,
        actor=actor_network,
        critic=critic_network,
        buffer=buffer,
    )

    algorithm = SAC(
        cfg=cfg,
        env=dummy_continuous_env,
        env_params=dummy_continuous_env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=alpha_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        buffer=buffer,
    )

    key, state = algorithm.init(rng_key)
    key, state = algorithm.warmup(key, state, num_steps=cfg.learning_starts)
    key, state, info = algorithm.train(key, state, num_steps=cfg.train_frequency)
    eval_key, eval_transitions = algorithm.evaluate(key, state, num_steps=2)
