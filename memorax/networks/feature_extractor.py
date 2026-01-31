from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class FeatureExtractor(nn.Module):
    observation_extractor: nn.Module
    action_extractor: Optional[nn.Module] = None
    reward_extractor: Optional[nn.Module] = None
    done_extractor: Optional[nn.Module] = None

    def extract(
        self,
        embeddings: dict,
        key: str,
        extractor: Optional[nn.Module],
        x: Optional[jnp.ndarray] = None,
    ):
        if extractor is not None and x is not None:
            embeddings[key] = extractor(x)

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        **kwargs,
    ):
        embeddings = {"observation": self.observation_extractor(observation)}
        self.extract(embeddings, "action_embedding", self.action_extractor, action)
        self.extract(embeddings, "reward_embedding", self.reward_extractor, reward)
        self.extract(
            embeddings, "done_embedding", self.done_extractor, done.astype(jnp.int32)
        )

        features = jnp.concatenate([*embeddings.values()], axis=-1)

        return features, embeddings
