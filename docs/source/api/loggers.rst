memorax.loggers
===============

Logging utilities for tracking training progress.

.. currentmodule:: memorax.loggers

Core
----

:class:`Logger` - Protocol that all loggers implement.

:class:`MultiLogger` - Composite logger that dispatches to multiple backends.

Backends
--------

:class:`DashboardLogger` - Rich terminal dashboard.

:class:`FileLogger` - Logs to file.

:class:`WandbLogger` - Weights & Biases integration.

:class:`TensorBoardLogger` - TensorBoard integration.

:class:`CheckpointLogger` - Orbax checkpoint saving.

.. autosummary::
   :toctree: generated
   :hidden:

   Logger
   MultiLogger
   CheckpointLogger
   DashboardLogger
   FileLogger
   WandbLogger
   TensorBoardLogger
