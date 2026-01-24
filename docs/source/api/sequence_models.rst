memorax.networks.sequence_models
================================

Sequence models for temporal processing in RL.

.. currentmodule:: memorax.networks.sequence_models

Base Class
----------

.. autoclass:: SequenceModel
   :members:
   :special-members: __init__, __call__

Wrappers
--------

.. autoclass:: SequenceModelWrapper
   :members:
   :special-members: __init__, __call__

.. autoclass:: MetaMaskWrapper
   :members:
   :special-members: __init__, __call__

RNNs
----

.. autoclass:: RNN
   :members:
   :special-members: __init__, __call__

.. autoclass:: sLSTMCell
   :members:
   :special-members: __init__, __call__

.. autoclass:: mLSTMCell
   :members:
   :special-members: __init__, __call__

State Space Models
------------------

.. autoclass:: LRUCell
   :members:
   :special-members: __init__, __call__

.. autoclass:: S5Cell
   :members:
   :special-members: __init__, __call__

.. autoclass:: MambaCell
   :members:
   :special-members: __init__, __call__

.. autoclass:: MinGRUCell
   :members:
   :special-members: __init__, __call__

Memory Models
-------------

.. autoclass:: FFMCell
   :members:
   :special-members: __init__, __call__

.. autoclass:: SHMCell
   :members:
   :special-members: __init__, __call__

.. autoclass:: Memoroid
   :members:
   :special-members: __init__, __call__

.. autoclass:: MemoroidCellBase
   :members:
   :special-members: __init__, __call__

Attention
---------

.. autoclass:: SelfAttention
   :members:
   :special-members: __init__, __call__

.. autoclass:: LinearAttentionCell
   :members:
   :special-members: __init__, __call__
