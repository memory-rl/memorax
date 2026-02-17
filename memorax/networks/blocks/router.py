import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


class TopKRouter(nn.Module):
    """Top-K router for Mixture of Experts."""

    num_experts: int
    k: int = 2

    @nn.compact
    def __call__(self, inputs: Array) -> tuple:
        batch_size, seq_len, _ = inputs.shape
        num_tokens = batch_size * seq_len

        logits = nn.Dense(self.num_experts, use_bias=False)(inputs)
        probs = jax.nn.softmax(logits, axis=-1)

        top_k_weights, top_k_indices = jax.lax.top_k(probs, self.k)
        weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-9)

        mask = jax.nn.one_hot(top_k_indices, self.num_experts).sum(axis=-2)
        mask = jnp.minimum(mask, 1.0)
        fraction = mask.sum(axis=(0, 1)) / num_tokens

        loss = jnp.mean(
            jnp.square(jax.nn.logsumexp(logits, axis=-1))
        ) + self.num_experts * jnp.sum(fraction * probs.mean(axis=(0, 1)))

        self.sow(
            "intermediates",
            "moe_loss",
            loss,
        )

        return weights, top_k_indices
