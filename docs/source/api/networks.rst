memorax.networks
================

Neural network components for building RL agents.

.. currentmodule:: memorax.networks

Network
-------

The main network class that composes feature extractors, torsos, and heads.

.. autoclass:: Network
   :members:
   :special-members: __init__, __call__

Feature Extraction
------------------

.. autoclass:: FeatureExtractor
   :members:
   :special-members: __init__, __call__

.. autoclass:: Identity
   :members:

Architectures
-------------

.. autoclass:: MLP
   :members:
   :special-members: __init__, __call__

.. autoclass:: CNN
   :members:
   :special-members: __init__, __call__

.. autoclass:: ViT
   :members:
   :special-members: __init__, __call__

Sequence Model Wrapper
----------------------

.. autoclass:: SequenceModelWrapper
   :members:
   :special-members: __init__, __call__
