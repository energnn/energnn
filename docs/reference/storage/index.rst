=======
Storage
=======

.. currentmodule:: energnn.storage

During a project's lifespan, multiple training runs are launched on various nodes,
each with a separate local storage.
It is thus essential to define remote storage as something common to a project.

The :class:`Storage` class serves as a link between the local storage and this common remote storage.
Multiple implementations are possible, but all should respect the following interface.


.. autoclass:: Storage
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    Storage.upload
    Storage.download

Implementation
==============

.. autoclass:: DummyStorage
   :no-members:
   :show-inheritance:
