memorax.networks.blocks
=======================

Building blocks for neural network architectures.

.. currentmodule:: memorax.networks.blocks

Feed-Forward
------------

.. autoclass:: FFN
   :members:
   :special-members: __init__, __call__

Mixture of Experts
------------------

.. autoclass:: MoE
   :members:
   :special-members: __init__, __call__

.. autoclass:: TopKRouter
   :members:
   :special-members: __init__, __call__

Normalization
-------------

.. autoclass:: PreNorm
   :members:
   :special-members: __init__, __call__

.. autoclass:: PostNorm
   :members:
   :special-members: __init__, __call__

Residual Connections
--------------------

.. autoclass:: Residual
   :members:
   :special-members: __init__, __call__

.. autoclass:: GatedResidual
   :members:
   :special-members: __init__, __call__

Composition
-----------

.. autoclass:: Stack
   :members:
   :special-members: __init__, __call__

.. autoclass:: SegmentRecurrence
   :members:
   :special-members: __init__, __call__
