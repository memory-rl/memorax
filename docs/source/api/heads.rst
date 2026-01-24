memorax.networks.heads
======================

Output heads for different RL objectives.

.. currentmodule:: memorax.networks.heads

Policy Heads
------------

:class:`Categorical` - Categorical policy for discrete actions.

:class:`Gaussian` - Gaussian policy for continuous actions.

:class:`SquashedGaussian` - Squashed Gaussian policy (tanh-bounded).

Value Heads
-----------

:class:`VNetwork` - State value function head.

:class:`DiscreteQNetwork` - Q-network for discrete actions.

:class:`ContinuousQNetwork` - Q-network for continuous actions.

Temperature
-----------

:class:`Alpha` - Learnable temperature parameter for SAC.

.. autosummary::
   :toctree: generated
   :hidden:

   Categorical
   Gaussian
   SquashedGaussian
   VNetwork
   DiscreteQNetwork
   ContinuousQNetwork
   Alpha
