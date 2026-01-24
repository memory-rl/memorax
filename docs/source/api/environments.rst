memorax.environments
====================

Environment creation and wrappers.

.. currentmodule:: memorax.environments

:func:`environment.make` - Factory function to create JAX-compatible environments.

Supported Environments
----------------------

The ``make`` function supports the following namespaces:

- ``gymnax::`` - Gymnax environments (CartPole, Pendulum, etc.)
- ``brax::`` - Brax physics environments (Ant, Humanoid, etc.)
- ``navix::`` - Navigation environments
- ``craftax::`` - Craftax multi-task environments
- ``popgym_arcade::`` - POPGym Arcade environments
- ``popjym::`` - POPJym memory benchmarks
- ``mujoco::`` - MuJoCo environments
- ``xminigrid::`` - XMiniGrid environments

.. autosummary::
   :toctree: generated
   :hidden:

   environment.make
