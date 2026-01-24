memorax.environments
====================

Environment creation and wrappers.

.. currentmodule:: memorax.environments

Factory Function
----------------

.. autofunction:: make

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
- ``gxm::`` - GXM environments
- ``xminigrid::`` - XMiniGrid environments

Example
-------

.. code-block:: python

   from memorax.environments import make

   # Gymnax environment
   env, env_params = make("gymnax::CartPole-v1")

   # Brax environment
   env, env_params = make("brax::ant")

   # POPGym environment
   env, env_params = make("popjym::RepeatPrevious-v0")
