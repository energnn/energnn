=======
Tracker
=======

.. currentmodule:: energnn.tracker


A typical EnerGNN project involves multiple training runs, and various datasets.
A :class:`Tracker` aims at versionning datasets and models, monitoring training runs,
and finding the best model.

Multiple implementations are possible, but all should respect the following
interface.

.. autoclass:: Tracker
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    Tracker.init_run
    Tracker.stop_run
    Tracker.get_amortizer_path
    Tracker.run_track_dataset
    Tracker.run_track_amortizer
    Tracker.run_append

Implementation
==============

.. autoclass:: DummyTracker
   :no-members:
   :show-inheritance:

.. autoclass:: NeptuneTracker
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    NeptuneTracker.init_run
    NeptuneTracker.stop_run
    NeptuneTracker.get_amortizer_path
    NeptuneTracker.run_track_dataset
    NeptuneTracker.run_track_amortizer
    NeptuneTracker.run_append

.. autoclass:: NeptuneScaleTracker
    :no-members:
    :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    NeptuneScaleTracker.init_run
    NeptuneScaleTracker.stop_run
    NeptuneScaleTracker.get_amortizer_path
    NeptuneScaleTracker.run_track_dataset
    NeptuneScaleTracker.run_track_amortizer
    NeptuneScaleTracker.run_append