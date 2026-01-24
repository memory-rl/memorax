memorax.loggers
===============

Logging utilities for tracking training progress.

.. currentmodule:: memorax.loggers

Core
----

:class:`Logger` - Composite logger that dispatches to multiple backends.

:class:`LoggerState` - State container for logger.

Backends
--------

:class:`ConsoleLogger` - Logs to console/stdout.

:class:`DashboardLogger` - Rich terminal dashboard.

:class:`FileLogger` - Logs to file.

:class:`WandbLogger` - Weights & Biases integration.

:class:`TensorBoardLogger` - TensorBoard integration.

:class:`NeptuneLogger` - Neptune.ai integration.

.. autosummary::
   :toctree: generated
   :hidden:

   Logger
   LoggerState
   ConsoleLogger
   DashboardLogger
   FileLogger
   WandbLogger
   TensorBoardLogger
   NeptuneLogger
