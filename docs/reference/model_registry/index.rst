==============
Model registry
==============

.. currentmodule:: energnn.model_registry

A **model registry** must be defined in order to use a :class:`SimpleAmortizer` train method.
It is used by the trainer to store and reference trainer snapshots during the training process, in order to reuse them after.

A valid implementation of :class:'ModelRegistry' must respect the following interface :


.. autoclass:: ModelRegistry
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    ModelRegistry.register_trainer
    ModelRegistry.get_trainer_metadata
    ModelRegistry.download_trainer
