import jax.numpy as jnp
import jaxmarl


def make(env_id, **kwargs):
    env = jaxmarl.make(env_id, **kwargs)
    return env, None


class JaxMarlWrapper:
    """Wraps a JAXMarl environment to return stacked arrays instead of per-agent dicts.

    This simplifies multi-agent algorithm implementations by providing a consistent
    array-based interface where the first axis is the agent dimension.
    """

    def __init__(self, env, agent_ids: tuple | None = None):
        self._env = env
        self._agent_ids = tuple(env.agents) if agent_ids is None else agent_ids

    @property
    def unwrapped(self):
        return self._env

    @property
    def agents(self):
        return self._agent_ids

    @property
    def num_agents(self):
        return len(self._agent_ids)

    @property
    def action_spaces(self):
        return self._env.action_spaces

    def reset(self, key):
        obs_dict, state = self._env.reset(key)
        obs = jnp.stack([obs_dict[aid] for aid in self._agent_ids])
        return obs, state

    def step(self, key, state, actions):
        actions_dict = {aid: actions[i] for i, aid in enumerate(self._agent_ids)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(
            key, state, actions_dict
        )
        return (
            jnp.stack([obs_dict[aid] for aid in self._agent_ids]),
            state,
            jnp.stack([reward_dict[aid] for aid in self._agent_ids]),
            jnp.stack([done_dict[aid] for aid in self._agent_ids]),
            info,
        )
