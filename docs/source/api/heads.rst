memorax.networks.heads
======================

Output heads for different RL objectives.

.. currentmodule:: memorax.networks.heads

Q-Networks
----------

.. autoclass:: DiscreteQNetwork
   :members:
   :special-members: __init__, __call__

.. autoclass:: ContinuousQNetwork
   :members:
   :special-members: __init__, __call__

Value Networks
--------------

.. autoclass:: VNetwork
   :members:
   :special-members: __init__, __call__

Policy Heads
------------

.. autoclass:: Categorical
   :members:
   :special-members: __init__, __call__

.. autoclass:: Gaussian
   :members:
   :special-members: __init__, __call__

.. autoclass:: SquashedGaussian
   :members:
   :special-members: __init__, __call__

Temperature
-----------

.. autoclass:: Alpha
   :members:
   :special-members: __init__, __call__
