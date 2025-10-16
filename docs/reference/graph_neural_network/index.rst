====================
Graph Neural Network
====================

.. currentmodule:: energnn.gnn

This package relies on the training of Graph Neural Networks
(GNNs), that take as input a context graph :math:`x`
to produce a meaningful prediction :math:`\hat{y}`.
The GNN is parameterized by a vector
parameter :math:`\theta` and is
denoted by :math:`\hat{y}_\theta`.
We choose to decompose our GNN architectures as follows:

.. math::
    \hat{y}_\theta = D_\theta \circ C_\theta \circ E_\theta.

- The *encoder* :math:`E_\theta` embeds input features
  into an abstract latent space.
- The *coupler* :math:`C_\theta` associates addresses
  with latent coordinates that
  reflect a coupling induced by the context graph.
- The *decoder* :math:`D_\theta` converts
  addresses latent coordinates into a meaningful
  prediction :math:`\hat{y}`.

In the case where :math:`\hat{y}` should be a graph
bearing a vector at every object
present in the context graph :math:`x`, then the GNN is
said to be :ref:`permutation-equivariant <equivariant>`.
In the case where :math:`\hat{y}` should be a vector
quantity common to the whole context graph :math:`x`,
then the GNN is
said to be :ref:`permutation-invariant <invariant>`.


.. _equivariant:

Equivariant GNN
===============

The main objective of this package is to deal with
the following learning problem:

.. math::
    \underset{\theta}{\min} \mathbb{E}_{x \sim p}
    \left[ f(\hat{y}_\theta(x); x) \right].

In this case, the quantity :math:`\hat{y}` is a graph,
with one vector per object present in the context graph
:math:`x`.
The object ordering in :math:`\hat{y}` should match
the one of :math:`x`.
In that sense, the GNN :math:`\hat{y}_\theta` should
be **permutation-equivariant**.

.. autoclass:: EquivariantGNN
  :no-members:
  :show-inheritance:

.. autosummary::
  :toctree: _autosummary
  :nosignatures:


.. _invariant:

Invariant GNN
=============

In some cases, it can be useful to
predict a scalar quantity :math:`\hat{y}`
that is common to the whole context graph :math:`x`.
This quantity should not depend on the
object ordering of context :math:`x`.
Therefore, the GNN :math:`\hat{y}_\theta` should
be **permutation-invariant**.

.. autoclass:: InvariantGNN
  :no-members:
  :show-inheritance:

.. autosummary::
  :toctree: _autosummary
  :nosignatures:


Components
==========

.. toctree::
   :maxdepth: 1

   encoder
   coupler/index
   decoder/equivariant
   decoder/invariant
   utils
