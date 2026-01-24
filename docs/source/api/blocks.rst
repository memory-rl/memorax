memorax.networks.blocks
=======================

Building blocks for constructing network architectures.

.. currentmodule:: memorax.networks

Feed-Forward
------------

:class:`FFN` - Feed-forward network block with expansion.

Normalization
-------------

:class:`PreNorm` - Pre-normalization wrapper.

:class:`PostNorm` - Post-normalization wrapper.

Residual
--------

:class:`Residual` - Residual connection wrapper.

Composition
-----------

:class:`Stack` - Stacks multiple blocks sequentially.

Mixture of Experts
------------------

:class:`MoE` - Mixture of Experts layer.

:class:`TopKRouter` - Top-K routing for MoE.

.. autosummary::
   :toctree: generated
   :hidden:

   FFN
   PreNorm
   PostNorm
   Residual
   Stack
   MoE
   TopKRouter
