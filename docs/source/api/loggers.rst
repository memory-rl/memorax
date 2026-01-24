memorax.loggers
===============

Logging utilities for experiment tracking.

.. currentmodule:: memorax.loggers

Logger
------

.. autoclass:: Logger
   :members:
   :special-members: __init__

.. autoclass:: LoggerState
   :members:

Logger Implementations
----------------------

.. autoclass:: ConsoleLogger
   :members:
   :special-members: __init__

.. autoclass:: DashboardLogger
   :members:
   :special-members: __init__

.. autoclass:: WandbLogger
   :members:
   :special-members: __init__

.. autoclass:: TensorBoardLogger
   :members:
   :special-members: __init__

.. autoclass:: NeptuneLogger
   :members:
   :special-members: __init__

.. autoclass:: FileLogger
   :members:
   :special-members: __init__

Example
-------

.. code-block:: python

   from memorax.loggers import Logger, WandbLogger, ConsoleLogger

   # Create composite logger
   logger = Logger([WandbLogger(), ConsoleLogger()])

   # Initialize
   logger_state = logger.init(cfg=config)

   # Log metrics
   logger_state = logger.log(logger_state, {"loss": 0.5}, step=100)

   # Emit buffered metrics
   logger.emit(logger_state)

   # Cleanup
   logger.finish(logger_state)
