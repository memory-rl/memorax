def make(env_id, **kwargs):
    import grimax
    from grimax.wrappers import AutoResetWrapper, GrimaxGymnaxWrapper

    env = grimax.make(env_id, **kwargs)
    params = env.default_params
    assert params.num_agents == 1, (
        f"Only single-agent grimax environments are supported, "
        f"got num_agents={params.num_agents}"
    )
    env = AutoResetWrapper(env)
    env = GrimaxGymnaxWrapper(env)

    return env, params
