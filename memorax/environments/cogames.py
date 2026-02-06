"""CoGames environment registration for PufferLib."""

from functools import partial
from typing import Optional, Sequence

import pufferlib.vector
from cogames.cli.mission import get_mission
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator

from memorax.environments.pufferlib import (
    PufferLibMultiAgentWrapper,
    PufferLibWrapper,
    register,
)


def _make_single(env_id: str, variants: Optional[Sequence[str]] = None, **kwargs):
    """Create a single MettaGrid environment."""
    _, cfg, _ = get_mission(env_id)

    if variants:
        try:
            from cogames.cogs_vs_clips.cogsguard_reward_variants import (
                apply_reward_variants,
            )

            apply_reward_variants(cfg, variants=variants)
        except ImportError:
            pass

    simulator = Simulator()
    return MettaGridPufferEnv(simulator, cfg, **kwargs)


def _cogames_factory(
    env_id: str,
    num_envs: int,
    variants: Optional[Sequence[str]] = None,
    multi_agent: bool = False,
    backend=pufferlib.vector.Serial,
    **kwargs,
):
    """Factory registered under the 'cogames' namespace."""
    _, cfg, _ = get_mission(env_id)
    num_agents = cfg.game.num_agents

    puffer_env = pufferlib.vector.make(
        partial(_make_single, variants=variants),
        num_envs=num_envs,
        backend=backend,
        env_kwargs={"env_id": env_id},
    )

    if multi_agent:
        wrapper = PufferLibMultiAgentWrapper(puffer_env, num_agents=num_agents)
    else:
        wrapper = PufferLibWrapper(puffer_env)

    env_info = {
        "agents_per_env": num_agents,
    }

    return wrapper, env_info


register("cogames", _cogames_factory)
