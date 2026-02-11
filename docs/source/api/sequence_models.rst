memorax.networks.sequence_models
================================

Sequence models for temporal processing.

.. currentmodule:: memorax.networks

RNN Models
----------

:class:`RNN` - Wrapper for Flax RNN cells (LSTM, GRU, etc.).

:class:`sLSTMCell` - Scalar LSTM cell with enhanced gating.

:class:`SHMCell` - Stable Hadamard Memory cell.

Memoroid Models
---------------

:class:`Memoroid` - Wrapper for parallel-scannable sequence models.

:class:`MemoroidCellBase` - Base class for memoroid cells.

:class:`MambaCell` - Selective State Space Model cell.

:class:`S5Cell` - Simplified Structured State Space cell.

:class:`LRUCell` - Linear Recurrent Unit cell.

:class:`MinGRUCell` - Minimal GRU cell.

:class:`mLSTMCell` - Matrix LSTM cell.

:class:`FFMCell` - Fast and Forgetful Memory cell.

:class:`LinearAttentionCell` - Linear attention cell.

Attention
---------

:class:`SelfAttention` - Multi-head self-attention.

Wrappers
--------

:class:`SequenceModelWrapper` - Wraps non-recurrent models.

:class:`RL2Wrapper` - RL² wrapper that preserves hidden state across episode boundaries within a trial.

.. autosummary::
   :toctree: generated
   :hidden:

   RNN
   sLSTMCell
   SHMCell
   Memoroid
   MemoroidCellBase
   MambaCell
   S5Cell
   LRUCell
   MinGRUCell
   mLSTMCell
   FFMCell
   LinearAttentionCell
   SelfAttention
   SequenceModelWrapper
   RL2Wrapper
