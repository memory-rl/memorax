memorax.networks
================

Neural network components for building RL agents.

.. currentmodule:: memorax.networks

Core
----

:class:`Network` - Main network class composing feature extractors, torsos, and heads.

:class:`FeatureExtractor` - Extracts features from observations, actions, rewards, and done flags.

:class:`Identity` - Identity module that passes input through unchanged.

Architectures
-------------

:class:`MLP` - Multi-layer perceptron.

:class:`CNN` - Convolutional neural network.

:class:`ViT` - Vision Transformer.

Wrappers
--------

:class:`SequenceModelWrapper` - Wraps non-recurrent models for use as sequence models.

.. autosummary::
   :toctree: generated
   :hidden:

   Network
   FeatureExtractor
   Identity
   MLP
   CNN
   ViT
   SequenceModelWrapper
