import jax

from memory_rl.algorithms import PPO, PPOConfig


def test_ppo_init_and_train(
    rng_key: jax.Array,
    dummy_env,
    dummy_env_params,
    ppo_components,
):
    actor, critic, optimizer = ppo_components
    cfg = PPOConfig(
        name="ppo-smoke",
        learning_rate=1e-3,
        num_envs=2,
        num_eval_envs=1,
        num_steps=2,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=1,
        update_epochs=1,
        normalize_advantage=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_starts=0,
        actor=actor,
        critic=critic,
    )

    algorithm = PPO(
        cfg=cfg,
        env=dummy_env,
        env_params=dummy_env_params,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )

    key, state = algorithm.init(rng_key)
    key, state = algorithm.warmup(key, state, num_steps=cfg.batch_size)
    rollout_steps = cfg.num_envs * cfg.num_steps
    key, state, transitions = algorithm.train(key, state, num_steps=rollout_steps)

    eval_key, eval_transitions = algorithm.evaluate(key, state, num_steps=2)
