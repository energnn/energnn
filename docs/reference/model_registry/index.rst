==============
Trainer registry
==============

.. currentmodule:: energnn.trainer_registry

A **trainer registry** must be defined in order to use a :class:`SimpleAmortizer` train method.
It is used by the trainer to store and reference trainer snapshots during the training process, in order to reuse them after.

A valid implementation of :class:'TrainerRegistry' must respect the following interface :


.. autoclass:: TrainerRegistry
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    TrainerRegistry.register_trainer
    TrainerRegistry.get_trainer_metadata
    TrainerRegistry.download_trainer


The :class:'LocalRegistry' is a local implementation of a :class:'TrainerRegistry', that stores the trainers in a local directory.

.. autoclass:: LocalRegistry
    :no-members:
    :show-inheritance: