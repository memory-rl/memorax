import jax
import jax.numpy as jnp


def memory_metrics(carry, initial_carry=None):
    """Compute model-agnostic memory metrics from any carry pytree.

    Uses jax.tree.leaves to flatten arbitrary carry structures (tuples,
    structs, nested dicts) and computes norms on each leaf array.
    Handles complex-valued carries (LRU, S5) via jnp.abs.

    Returns dict with:
        memory/carry_norm: mean L2 norm across carry leaves
        memory/carry_delta: mean L2 norm of (carry - initial_carry) per leaf
    """
    if carry is None:
        metrics = {"memory/carry_norm": jnp.float32(0.0)}
        if initial_carry is not None:
            metrics["memory/carry_delta"] = jnp.float32(0.0)
        return metrics

    leaves = jax.tree.leaves(carry)

    norms = jnp.stack(
        [jnp.linalg.norm(jnp.abs(leaf).astype(jnp.float32).reshape(-1)) for leaf in leaves]
    )

    metrics = {"memory/carry_norm": jnp.mean(norms)}

    if initial_carry is not None:
        initial_leaves = jax.tree.leaves(initial_carry)
        deltas = jnp.stack(
            [
                jnp.linalg.norm(jnp.abs(leaf - prev).astype(jnp.float32).reshape(-1))
                for leaf, prev in zip(leaves, initial_leaves)
            ]
        )
        metrics["memory/carry_delta"] = jnp.mean(deltas)

    return metrics
