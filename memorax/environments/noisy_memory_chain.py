from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    context: int
    query: int
    total_perfect: int
    total_regret: float
    time: int
    current_noise: jax.Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    memory_length: int = 5
    max_steps_in_episode: int = 1000


class NoisyMemoryChain(environment.Environment[EnvState, EnvParams]):

    def __init__(self, num_bits: int = 1, n_noise: int = 10):
        super().__init__()
        self.num_bits = num_bits
        self.n_noise = n_noise

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        key, key_noise = jax.random.split(key)

        obs = self.get_obs(state, params)

        reward = 0.0
        mem_not_full = state.time < params.memory_length
        correct_action = action == state.context[state.query]
        mem_correct = jnp.logical_and(1 - mem_not_full, correct_action)
        mem_wrong = jnp.logical_and(1 - mem_not_full, 1 - correct_action)
        reward = reward + mem_correct - mem_wrong

        new_noise = jax.random.normal(key_noise, shape=(self.n_noise,))

        state = EnvState(
            context=jnp.int32(state.context),
            query=jnp.int32(state.query),
            total_perfect=jnp.int32(state.total_perfect + mem_correct),
            total_regret=jnp.float32(state.total_regret + 2 * mem_wrong),
            time=state.time + 1,
            current_noise=new_noise,
        )

        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key_context, key_query, key_noise = jax.random.split(key, 3)
        context = jax.random.bernoulli(key_context, p=0.5, shape=(self.num_bits,))
        query = jax.random.randint(key_query, minval=0, maxval=self.num_bits, shape=())
        noise = jax.random.normal(key_noise, shape=(self.n_noise,))

        state = EnvState(
            context=jnp.int32(context),
            query=jnp.int32(query),
            total_perfect=0,
            total_regret=jnp.float32(0),
            time=0,
            current_noise=noise,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        obs = jnp.zeros(shape=(self.num_bits + 2,), dtype=jnp.float32)

        obs = obs.at[0].set(
            1 - state.time / params.memory_length,
        )

        query_val = jax.lax.select(
            state.time == params.memory_length - 1, state.query, 0
        )
        obs = obs.at[1].set(query_val)

        context_val = jax.lax.select(
            state.time == 0, (2 * state.context - 1).squeeze(), 0
        )
        obs = obs.at[2:].set(context_val)

        return jnp.concatenate([obs, state.current_noise])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        done_mem = state.time - 1 == params.memory_length
        return jnp.logical_or(done_steps, done_mem)

    @property
    def name(self) -> str:
        return "NoisyMemoryChain-bsuite"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0,
            2 * self.num_bits,
            (self.num_bits + 2 + self.n_noise,),
            jnp.float32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "context": spaces.Discrete(2),
                "query": spaces.Discrete(self.num_bits),
                "total_perfect": spaces.Discrete(params.max_steps_in_episode),
                "total_regret": spaces.Discrete(params.max_steps_in_episode),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "current_noise": spaces.Box(
                    -jnp.inf, jnp.inf, (self.n_noise,), jnp.float32
                ),
            }
        )
