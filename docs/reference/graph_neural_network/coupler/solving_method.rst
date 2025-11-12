========================
Solving method :math:`S`
========================

A solving method is a functional that processes a coupling function into address latent coordinates.

The solving method processes the coupling function
:math:`h \mapsto F_\theta(h;x)` parameterized by the context graph :math:`x` into unique address coordinates.

Multiple implementations are possible, but all should respect the provided interface.


.. currentmodule:: energnn.gnn.coupler.solving_method

.. autoclass:: SolvingMethod
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    SolvingMethod.initialize_coordinates
    SolvingMethod.solve


Implementations
===============

.. autoclass:: ZeroSolvingMethod
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    ZeroSolvingMethod.initialize_coordinates
    ZeroSolvingMethod.solve



.. autoclass:: NeuralODESolvingMethod
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    NeuralODESolvingMethod.initialize_coordinates
    NeuralODESolvingMethod.solve
