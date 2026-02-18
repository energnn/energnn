=========
Amortizer
=========

.. currentmodule:: energnn.amortizer

.. image:: ../../_static/energnn-pipeline-black.png
   :align: center
   :width: 500
   :class: only-light

.. image:: ../../_static/energnn-pipeline-white.png
   :align: center
   :width: 500
   :class: only-dark


An **amortizer** contains the Graph Neural Network :math:`\hat{y}_\theta` and the pre-processing
and post-processing layers, and can be trained to solve the following amortized
optimization problem,

.. math::
    \underset{\theta}{\min} \mathbb{E}_{x \sim p} \left[ f(\hat{y}_\theta(x);x) \right].


.. autoclass:: SimpleAmortizer
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    SimpleAmortizer.init
    SimpleAmortizer.train
    SimpleAmortizer.run_evaluation
    SimpleAmortizer.eval
    SimpleAmortizer.training_step
    SimpleAmortizer.eval_step
    SimpleAmortizer.forward_batch
    SimpleAmortizer.infer
    SimpleAmortizer.infer_batch
    SimpleAmortizer._apply_model
    SimpleAmortizer.forward
    SimpleAmortizer.update_params
    SimpleAmortizer.save
    SimpleAmortizer.load
    

Utility classes and functions for Amortizer
===========================================

The :class:`SimpleAmortizer` class used some class and functions in the training process for 
task logging and metrics collection.

Task logging
------------

.. autoclass:: TaskLogger
    :no-members:
    :show-inheritance:

Metrics collection
------------------
    
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    append_metrics_and_infos
    numpify_info_dict