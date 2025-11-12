=======
Problem
=======

In this package, we assume the following initial optimization problem definition,

.. math:: 
 \underset{y}{\min} \ f (y; x).

This optimization problem is defined by :

- an objective function :math:`f`  **shared across a problem class**;
- a context :math:`x` **specific to each problem instance**.

Moreover, it is assumed that the gradient :math:`\nabla_y f` is defined.

Every class of problem differs from one another.
This package provides an interface that should be respected
for compatibility with our neural network library.



.. currentmodule:: energnn.problem


Problem
=======

.. autoclass:: Problem
   :no-members:
   :show-inheritance:
   
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Problem.__init__
   Problem.get_context
   Problem.get_zero_decision
   Problem.get_gradient
   Problem.get_metrics


Batch
=====

.. autoclass:: ProblemBatch
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ProblemBatch.__init__
   ProblemBatch.get_context
   ProblemBatch.get_zero_decision
   ProblemBatch.get_gradient
   ProblemBatch.get_metrics
   ProblemBatch.get_decision_structure


Dataset
=======

.. autoclass:: ProblemDataset
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ProblemDataset.get_infos_for_feature_store
   ProblemDataset.get_locally_missing_instances
   ProblemDataset.get_instance_paths
   ProblemDataset.to_json
   ProblemDataset.to_pickle
   ProblemDataset.from_pickle

Metadata
========

.. autoclass:: ProblemMetadata
   :no-members:
   :show-inheritance:

Loader
======

.. autoclass:: ProblemLoader
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ProblemLoader.__init__
   ProblemLoader.__iter__
   ProblemLoader.__next__
   ProblemLoader.__len__