import jax

from memory_rl.algorithms import PPOContinuous, PPOContinuousConfig


def test_ppo_continuous_init_and_train(
    rng_key: jax.Array,
    dummy_continuous_env,
    dummy_continuous_env_params,
    ppo_continuous_components,
):
    actor, critic, optimizer = ppo_continuous_components

    cfg = PPOContinuousConfig(
        name="ppo-continuous-smoke",
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

    algorithm = PPOContinuous(
        cfg=cfg,
        env=dummy_continuous_env,
        env_params=dummy_continuous_env_params,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )

    key, state = algorithm.init(rng_key)
    rollout_steps = cfg.num_envs * cfg.num_steps * 100
    key, state, transitions = algorithm.train(key, state, num_steps=rollout_steps)
    eval_key, eval_transitions = algorithm.evaluate(key, state, num_steps=2)
