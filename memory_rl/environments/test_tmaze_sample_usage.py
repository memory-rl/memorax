# this is pretty a buggy file i just used it as a quick smoke test so can be just removed at any time
# thought i leave it here as reference for now on how to use the tmaze_env

import jax
from memory_rl.environments.tmaze_env import make_tmaze_env


def pretty_obs(obs):
    obs_type_map = {0: "null", 1: "oracle", 2: "junction", 3: "goal"}
    goal_side_map = {-1: "Left", 1: "Right", -2: "Unknown"}
    obs_type = int(obs[0])
    goal_obs = int(obs[1])
    return f"[{obs_type} ({obs_type_map.get(obs_type, str(obs_type))}), {goal_obs} ({goal_side_map.get(goal_obs, str(goal_obs))})]"


def pretty_action(action):
    action_map = {0: "L", 1: "R", 2: "U", 3: "D"}
    return f"{action} ({action_map.get(action, str(action))})"


def print_trajectory(env, key, actions, label):
    timestep = env.reset(key)
    print(f"\n{label} TMaze trajectory (custom actions):")
    for step, action in enumerate(actions):
        obs = timestep.observation
        pos = (int(timestep.state.x), int(timestep.state.y))
        reward = timestep.reward
        done = bool(timestep.is_done())
        print(
            f"Step {step}: obs={pretty_obs(obs)}, pos={pos}, action={pretty_action(action)}, reward={reward}, done={done}"
        )
        if done:
            break
        timestep = env.step(timestep, action)


def run_comprehensive_tests():
    # Passive TMaze: deterministic policy to goal (goal_side=1)
    env_passive, key_passive = make_tmaze_env(L=5, active=False, seed=0)
    actions_to_goal_right = (
        [1] * 5 + [3] + [3]
    )  # Right to junction, Down to (L,1), Down again to stay at goal
    print_trajectory(
        env_passive,
        key_passive,
        actions_to_goal_right,
        "Passive (goal_side=1, reach goal)",
    )

    # Passive TMaze: deterministic policy to goal (goal_side=-1)
    env_passive_left, key_passive_left = make_tmaze_env(L=5, active=False, seed=123)
    actions_to_goal_left = (
        [1] * 5 + [2] + [2]
    )  # Right to junction, Up to (L,-1), Up again to stay at goal
    print_trajectory(
        env_passive_left,
        key_passive_left,
        actions_to_goal_left,
        "Passive (goal_side=-1, reach goal)",
    )

    # Active TMaze: visit oracle, then go to goal (goal_side=1)
    env_active, key_active = make_tmaze_env(L=5, active=True, seed=42)
    actions_active_right = (
        [0] + [1] * 6 + [3] + [3]
    )  # Left to oracle, right to junction (6 rights to reach x=5), down to (L,1), down again
    print_trajectory(
        env_active, key_active, actions_active_right, "Active (goal_side=1, reach goal)"
    )

    # Active TMaze: visit oracle, then go to goal (goal_side=-1)
    env_active_left, key_active_left = make_tmaze_env(L=5, active=True, seed=43)
    actions_active_left = (
        [0] + [1] * 6 + [2] + [2]
    )  # Left to oracle, right to junction (6 rights to reach x=5), up to (L,-1), up again
    print_trajectory(
        env_active_left,
        key_active_left,
        actions_active_left,
        "Active (goal_side=-1, reach goal)",
    )

    # Random walk for both variants
    import numpy as np

    for label, (env, key) in [
        ("Passive", (env_passive, key_passive)),
        ("Active", (env_active, key_active)),
    ]:
        np.random.seed(0)
        actions = np.random.choice([0, 1, 2, 3], size=10)
        print_trajectory(env, key, actions, f"{label} (random walk)")


def test_jit_compilation():
    """Test that the environment functions work with JAX JIT compilation."""
    print("\n=== Testing JAX JIT Compilation ===")

    # Test with passive TMaze
    env_passive, key_passive = make_tmaze_env(L=5, active=False, seed=0)

    # JIT compile the reset and step functions
    jit_reset = jax.jit(env_passive.reset)
    jit_step = jax.jit(env_passive.step)

    print("Testing JIT-compiled reset function...")
    timestep = jit_reset(key_passive)
    print(
        f"  Reset successful: obs={timestep.observation}, pos=({int(timestep.state.x)}, {int(timestep.state.y)})"
    )

    print("Testing JIT-compiled step function...")
    action = 1  # Right
    new_timestep = jit_step(timestep, action)
    print(
        f"  Step successful: obs={new_timestep.observation}, pos=({int(new_timestep.state.x)}, {int(new_timestep.state.y)}), reward={new_timestep.reward}"
    )

    # Test with active TMaze
    env_active, key_active = make_tmaze_env(L=5, active=True, seed=42)
    jit_reset_active = jax.jit(env_active.reset)
    jit_step_active = jax.jit(env_active.step)

    print("Testing JIT-compiled active TMaze...")
    timestep_active = jit_reset_active(key_active)
    new_timestep_active = jit_step_active(timestep_active, 0)  # Left
    print(
        f"  Active TMaze JIT successful: obs={new_timestep_active.observation}, pos=({int(new_timestep_active.state.x)}, {int(new_timestep_active.state.y)})"
    )

    print("JAX JIT compilation tests passed!")


if __name__ == "__main__":
    run_comprehensive_tests()
    test_jit_compilation()
