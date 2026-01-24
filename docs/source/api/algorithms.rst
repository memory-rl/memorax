memorax.algorithms
==================

Reinforcement learning algorithms for training agents.

.. currentmodule:: memorax.algorithms

Overview
--------

Memorax provides several RL algorithms, each with a Config, State, and Algorithm class.

PPO
---

Proximal Policy Optimization for discrete and continuous action spaces.

.. autoclass:: PPO
   :members:
   :special-members: __init__

.. autoclass:: PPOConfig
   :members:

.. autoclass:: PPOState
   :members:

DQN
---

Deep Q-Network with double and dueling variants.

.. autoclass:: DQN
   :members:
   :special-members: __init__

.. autoclass:: DQNConfig
   :members:

.. autoclass:: DQNState
   :members:

SAC
---

Soft Actor-Critic for continuous control.

.. autoclass:: SAC
   :members:
   :special-members: __init__

.. autoclass:: SACConfig
   :members:

.. autoclass:: SACState
   :members:

PQN
---

Policy Q-Network (on-policy Q-learning).

.. autoclass:: PQN
   :members:
   :special-members: __init__

.. autoclass:: PQNConfig
   :members:

.. autoclass:: PQNState
   :members:
