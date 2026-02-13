import jax
import jax.numpy as jnp


def memory_metrics(carry, initial_carry=None):
    if not jax.tree.leaves(carry):
        metrics = {"memory/carry_norm": jnp.float32(0.0)}
        if initial_carry is not None:
            metrics["memory/carry_delta"] = jnp.float32(0.0)
        return metrics

    leaves = jax.tree.leaves(carry)

    norms = jnp.stack(
        [
            jnp.linalg.norm(jnp.abs(leaf).astype(jnp.float32).reshape(-1))
            for leaf in leaves
        ]
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
