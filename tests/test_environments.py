"""Tests for all environment wrappers and the unified make interface.

Tests verify that environments properly implement:
- reset(key, params) -> (obs, state)
- step(key, state, action, params) -> (obs, state, reward, done, info)
- action_space(params) -> Space
- observation_space(params) -> Space
"""

import jax
import jax.numpy as jnp
import pytest


class TestGymnaxEnvironments:
    """Test gymnax environments via the memorax wrapper."""

    @pytest.fixture
    def cartpole(self):
        from memorax.environments import gymnax

        return gymnax.make("CartPole-v1")

    @pytest.fixture
    def pendulum(self):
        from memorax.environments import gymnax

        return gymnax.make("Pendulum-v1")

    @pytest.fixture
    def bsuite_catch(self):
        from memorax.environments import gymnax

        return gymnax.make("Catch-bsuite")

    def test_cartpole_reset(self, cartpole):
        env, env_params = cartpole
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None
        assert obs.shape == env.observation_space(env_params).shape

    def test_cartpole_step(self, cartpole):
        env, env_params = cartpole
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert next_state is not None
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(info, dict)

    def test_cartpole_action_space(self, cartpole):
        env, env_params = cartpole
        action_space = env.action_space(env_params)
        assert hasattr(action_space, "n")  # Discrete action space
        assert action_space.n == 2

    def test_cartpole_observation_space(self, cartpole):
        env, env_params = cartpole
        obs_space = env.observation_space(env_params)
        assert hasattr(obs_space, "shape")
        assert obs_space.shape == (4,)

    def test_pendulum_reset(self, pendulum):
        env, env_params = pendulum
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_pendulum_step(self, pendulum):
        env, env_params = pendulum
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()

    def test_pendulum_action_space(self, pendulum):
        env, env_params = pendulum
        action_space = env.action_space(env_params)
        assert hasattr(action_space, "shape")  # Continuous action space
        assert action_space.shape == (1,)

    def test_bsuite_wrapper(self, bsuite_catch):
        """Test that bsuite environments get the BSuiteWrapper."""
        env, env_params = bsuite_catch
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        # BSuiteWrapper adds extra state fields
        assert hasattr(state, "episode_regret")
        assert hasattr(state, "returned_episode_regret")
        assert hasattr(state, "timestep")

    def test_bsuite_step_info(self, bsuite_catch):
        """Test that bsuite step returns regret info."""
        env, env_params = bsuite_catch
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        _, _, _, _, info = env.step(key, state, action, env_params)

        assert "returned_episode_regret" in info
        assert "returned_episode" in info
        assert "timestep" in info
        assert "step_regret" in info
        assert "total_regret" in info


class TestUnifiedMakeInterface:
    """Test the unified make interface from memorax.environments."""

    def test_make_gymnax(self):
        from memorax.environments import make

        env, env_params = make("gymnax::CartPole-v1")
        assert env is not None
        assert env_params is not None

    def test_make_unknown_namespace(self):
        from memorax.environments import make

        with pytest.raises(ValueError, match="Unknown namespace"):
            make("unknown::SomeEnv")

    def test_make_parses_namespace(self):
        """Test that make correctly parses namespace::env_id format."""
        from memorax.environments import make

        # Should not raise
        env, _ = make("gymnax::Pendulum-v1")
        key = jax.random.key(0)
        obs, state = env.reset(key, _)
        assert obs is not None


class TestPopJymEnvironments:
    """Test PopJym environments."""

    @pytest.fixture
    def repeat_first(self):
        pytest.importorskip("popjym")
        from memorax.environments import popjym

        return popjym.make("RepeatFirstEasy")

    def test_reset(self, repeat_first):
        env, env_params = repeat_first
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_step(self, repeat_first):
        env, env_params = repeat_first
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()
        assert done.shape == ()

    def test_env_params_has_max_steps(self, repeat_first):
        _, env_params = repeat_first
        assert hasattr(env_params, "max_steps_in_episode")
        assert env_params.max_steps_in_episode == 52  # RepeatFirstEasy


class TestBraxEnvironments:
    """Test Brax environments (requires brax optional dependency)."""

    @pytest.fixture
    def ant_env(self):
        pytest.importorskip("brax")
        from memorax.environments import brax

        return brax.make("ant", mode="F")

    def test_reset(self, ant_env):
        env, env_params = ant_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_step(self, ant_env):
        env, env_params = ant_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        action_space = env.action_space(env_params)
        action = jnp.zeros(action_space.shape)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()

    def test_action_space(self, ant_env):
        env, env_params = ant_env
        action_space = env.action_space(env_params)
        assert hasattr(action_space, "shape")
        assert len(action_space.shape) == 1

    def test_observation_mask(self, ant_env):
        """Test that MaskObservationWrapper properly masks observations."""
        env, env_params = ant_env
        obs_space = env.observation_space(env_params)
        # Mode "F" should include all 27 dimensions for ant
        assert obs_space.shape == (27,)


class TestCraftaxEnvironments:
    """Test Craftax environments (requires craftax optional dependency)."""

    @pytest.fixture
    def craftax_env(self):
        pytest.importorskip("craftax")
        from memorax.environments import craftax

        return craftax.make("Craftax-Symbolic-v1", auto_reset=True)

    def test_reset(self, craftax_env):
        env, env_params = craftax_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_step(self, craftax_env):
        env, env_params = craftax_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()

    def test_env_params_has_max_steps(self, craftax_env):
        _, env_params = craftax_env
        assert hasattr(env_params, "max_steps_in_episode")


class TestNavixEnvironments:
    """Test Navix environments (requires navix optional dependency)."""

    @pytest.fixture
    def navix_env(self):
        pytest.importorskip("navix")
        from memorax.environments import navix

        return navix.make("Navix-Empty-5x5-v0")

    def test_reset(self, navix_env):
        env, _ = navix_env
        key = jax.random.key(0)
        obs, state = env.reset(key)
        assert obs is not None
        assert state is not None

    def test_step(self, navix_env):
        env, _ = navix_env
        key = jax.random.key(0)
        obs, state = env.reset(key)

        action = 0  # Move right
        next_obs, next_state, reward, done, info = env.step(key, state, action)

        assert next_obs is not None
        assert reward.shape == ()

    def test_action_space(self, navix_env):
        env, _ = navix_env
        action_space = env.action_space(None)
        assert hasattr(action_space, "n")  # Discrete

    def test_observation_space(self, navix_env):
        env, _ = navix_env
        obs_space = env.observation_space(None)
        assert hasattr(obs_space, "shape")


class TestXMinigridEnvironments:
    """Test XMinigrid environments (requires xminigrid optional dependency)."""

    @pytest.fixture
    def xminigrid_env(self):
        xminigrid = pytest.importorskip("xminigrid")
        from memorax.environments import xminigrid as xmg

        return xmg.make(
            "XLand-MiniGrid-R1-9x9",
            benchmark_name="trivial-21k",
            ruleset_key=jax.random.key(0),
        )

    @pytest.mark.xfail(
        reason="XMinigrid wrapper reset returns state without observation attribute"
    )
    def test_reset(self, xminigrid_env):
        env, env_params = xminigrid_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    @pytest.mark.xfail(
        reason="XMinigrid wrapper reset returns state without observation attribute"
    )
    def test_step(self, xminigrid_env):
        env, env_params = xminigrid_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()


class TestPopGymArcadeEnvironments:
    """Test PopGym Arcade environments (requires popgym_arcade optional dependency)."""

    @pytest.fixture
    def popgym_env(self):
        pytest.importorskip("popgym_arcade")
        from memorax.environments import popgym_arcade

        return popgym_arcade.make("RepeatFirst-easy")

    def test_reset(self, popgym_env):
        env, env_params = popgym_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_step(self, popgym_env):
        env, env_params = popgym_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action = env.action_space(env_params).sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None
        assert reward.shape == ()


class TestGxmEnvironments:
    """Test GXM environments (requires gxm optional dependency)."""

    @pytest.fixture
    def gxm_env(self):
        pytest.importorskip("gxm")
        from memorax.environments import gxm

        return gxm.make("Gymnax/CartPole-v1")

    def test_reset(self, gxm_env):
        env, env_params = gxm_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    def test_step(self, gxm_env):
        env, env_params = gxm_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        key, step_key = jax.random.split(key)
        action_space = env.action_space(env_params)
        action = action_space.sample(step_key)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None

    def test_action_space(self, gxm_env):
        env, env_params = gxm_env
        action_space = env.action_space(env_params)
        assert action_space is not None

    @pytest.mark.xfail(
        reason="GXM GymnaxEnvironment wrapper lacks observation_space attribute"
    )
    def test_observation_space(self, gxm_env):
        env, env_params = gxm_env
        obs_space = env.observation_space(env_params)
        assert obs_space is not None


class TestMujocoEnvironments:
    """Test MuJoCo environments (requires mujoco_playground optional dependency)."""

    @pytest.fixture
    def mujoco_env(self):
        pytest.importorskip("mujoco_playground")
        from memorax.environments import mujoco

        return mujoco.make("CheetahRun")

    @pytest.mark.xfail(
        reason="MuJoCo AutoResetWrapper expects pipeline_state attribute not present in State"
    )
    def test_reset(self, mujoco_env):
        env, env_params = mujoco_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)
        assert obs is not None
        assert state is not None

    @pytest.mark.xfail(
        reason="MuJoCo AutoResetWrapper expects pipeline_state attribute not present in State"
    )
    def test_step(self, mujoco_env):
        env, env_params = mujoco_env
        key = jax.random.key(0)
        obs, state = env.reset(key, env_params)

        action_space = env.action_space(env_params)
        action = jnp.zeros(action_space.shape)
        next_obs, next_state, reward, done, info = env.step(
            key, state, action, env_params
        )

        assert next_obs is not None

    def test_action_space(self, mujoco_env):
        env, env_params = mujoco_env
        action_space = env.action_space(env_params)
        assert hasattr(action_space, "shape")


class TestVmapCompatibility:
    """Test that environments work with JAX vmap for parallel execution."""

    def test_gymnax_vmap_reset(self):
        from memorax.environments import gymnax

        env, env_params = gymnax.make("CartPole-v1")

        num_envs = 4
        keys = jax.random.split(jax.random.key(0), num_envs)
        obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

        assert obs.shape == (num_envs, 4)

    def test_gymnax_vmap_step(self):
        from memorax.environments import gymnax

        env, env_params = gymnax.make("CartPole-v1")

        num_envs = 4
        keys = jax.random.split(jax.random.key(0), num_envs)
        obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

        step_keys = jax.random.split(jax.random.key(1), num_envs)
        actions = jnp.zeros(num_envs, dtype=jnp.int32)

        next_obs, next_states, rewards, dones, infos = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, states, actions, env_params)

        assert next_obs.shape == (num_envs, 4)
        assert rewards.shape == (num_envs,)
        assert dones.shape == (num_envs,)
