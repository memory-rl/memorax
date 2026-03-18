# 🔪 Sharp Bits 🔪

Memorax supports both JAX-native environments (gymnax, craftax, navix, etc.) and CPU callback-based environments (gymnasium, atari, pufferlib). The callback environments bridge external CPU-side envs into JAX via `jax.pure_callback`, which introduces several gotchas.

## CPU callback environments are opaque

The gymnasium, atari, and pufferlib wrappers use `jax.pure_callback` to bridge external CPU environments into JAX. The env state (`GymnasiumState`, `AtariState`, etc.) only tracks a step counter — the actual environment state lives in the external process. You cannot inspect, copy, or fork it.

This rules out algorithms that need to clone or restore env state, such as MCTS or other planning methods.

## `reset` resets the real environment

Calling `reset` on a callback env resets the underlying CPU environment. If you call `evaluate` (which resets) during training, you destroy the training environment's state.

Unlike JAX-native envs where `reset` creates a fresh state without affecting others, callback envs have a single shared mutable environment behind the callback.

```python
env, env_params = make("gymnasium::CartPole-v1")

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = agent.init(init_key)
key, train_key = jax.random.split(key)
state = agent.train(train_key, state, num_steps=10_000)

key, eval_key = jax.random.split(key)
state = agent.evaluate(eval_key, state)
```

## No `jax.vmap` over seeds with callback envs

The algorithms internally `vmap` over environments, so multi-env training works fine out of the box. However, you cannot wrap `init`/`train` with `jax.vmap` over multiple seeds like you can with JAX-native envs, since the callback env is a single shared external process.

```python
env, env_params = make("gymnax::CartPole-v1")

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
key = jax.random.key(0)
init = jax.vmap(agent.init)
train = jax.vmap(agent.train, in_axes=(0, 0, None))
key, init_key = jax.random.split(key)
states = init(jax.random.split(init_key, 4))
key, train_key = jax.random.split(key)
states = train(jax.random.split(train_key, 4), states, 10_000)
```

## `env_params` is always `None`

Callback envs don't use gymnax-style env params. Configuration happens at env construction time. The `env_params` returned by `make` will be `None`.
