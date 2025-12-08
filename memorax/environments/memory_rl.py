from memorax.environments.noisy_memory_chain import NoisyMemoryChain


def make(env_id, **kwargs):
    match env_id:
        case "NoisyMemoryChain-bsuite":
            env = NoisyMemoryChain(**kwargs)
            env_params = env.default_params
        case _:
            raise ValueError(f"Unknown env_id: {env_id}")

    return env, env_params
