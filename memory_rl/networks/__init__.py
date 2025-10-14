import memory_rl.networks.heads as heads
from memory_rl.networks.cnn import CNN
from memory_rl.networks.mlp import MLP
from memory_rl.networks.networks import Network, SequenceNetwork
from memory_rl.networks.recurrent import (
    FFMCell,
    GPT2,
    RNN,
    mLSTMBlock,
    sLSTMCell,
    sLSTMBlock,
    GTrXL,
    SHMCell,
    S5,
)
from memory_rl.networks.feature_extractors import (
    SeparateFeatureExtractor,
    SharedFeatureExtractor,
)
