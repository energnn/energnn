==========
Normalizer
==========

.. currentmodule:: energnn.normalizer

Normalizers serve as essential components in the training pipeline of neural networks,
as they are responsible for transforming input and output data to adhere to reasonable
distributions (*e.g.* zero mean and unitary standard deviation).
These distributions are crucial for the effective training of neural networks.
There are two types of normalizers.

- **Preprocessors --** These mappings are responsible for transforming input features to
  conform to reasonable distributions.

- **Postprocessors --** They scale the output of the neural network
  appropriately for the specific problem at hand. Additionally, postprocessors precondition
  the gradient stemming from the problem to make it better suited for the neural
  network's training.


One important requirement is that preprocessors and postprocessors must be **bijective** mappings.
Another key aspect of these mappings is that they need to be **permutation-equivariant** in order
to preserve the graph structure of the input :math:`x` and decision :math:`y`.
Both preprocessors and preprocessors are made of a series of class-specific and feature-specific
bijective functions, called **normalization functions**,
that can be fit to a provided :class:`energnn.problem.ProblemLoader`.

Preprocessor
============

.. autoclass:: Preprocessor
    :no-members:
    :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
    Preprocessor.fit_problem_loader
    Preprocessor.to_pickle
    Preprocessor.from_pickle
    Preprocessor.preprocess
    Preprocessor.preprocess_batch
    Preprocessor.preprocess_inverse
    Preprocessor.preprocess_inverse_batch

Postprocessor
=============

.. autoclass:: Postprocessor
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    Postprocessor.fit_problem_loader
    Postprocessor.to_pickle
    Postprocessor.from_pickle
    Postprocessor.postprocess
    Postprocessor.postprocess_batch
    Postprocessor.precondition_gradient
    Postprocessor.precondition_gradient_batch

Normalization Function Library
==============================

.. currentmodule:: energnn.normalizer.normalization_function

Multiple types and implementations of **normalization functions** are possible, but all
should respect the :class:`NormalizationFunction` interface.

.. autoclass:: NormalizationFunction
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    NormalizationFunction.init_aux
    NormalizationFunction.update_aux
    NormalizationFunction.compute_params
    NormalizationFunction.apply
    NormalizationFunction.apply_inverse
    NormalizationFunction.gradient_inverse

Identity
--------
.. autoclass:: IdentityFunction
    :no-members:
    :show-inheritance:

Center and Reduce
-----------------
.. autoclass:: CenterReduceFunction
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    CenterReduceFunction.init_aux
    CenterReduceFunction.update_aux
    CenterReduceFunction.compute_params
    CenterReduceFunction.apply
    CenterReduceFunction.apply_inverse
    CenterReduceFunction.gradient_inverse

Piecewise Linear Approximation of the Empirical CDF
---------------------------------------------------
.. autoclass:: CDFPWLinearFunction
    :no-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    CDFPWLinearFunction.init_aux
    CDFPWLinearFunction.update_aux
    CDFPWLinearFunction.compute_params
    CDFPWLinearFunction.apply
    CDFPWLinearFunction.apply_inverse
    CDFPWLinearFunction.gradient_inverse
