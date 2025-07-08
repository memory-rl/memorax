from hydra.utils import instantiate
import flax.linen as nn
from typing import Any

class Network(nn.Module):
    action_dim: int
    cfg: Any

    @nn.compact
    def __call__(self, x):
        x = instantiate(self.cfg.feature_extractor, layers=self.cfg.feature_extractor.layers)(x)
        x = instantiate(self.cfg.torso, layers=self.cfg.torso.layers)(x)
        x = instantiate(self.cfg.head, action_dim=self.action_dim)(x)
        return x

