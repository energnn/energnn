===========================================
Equivariant Decoder :math:`D^{equi}_\theta`
===========================================

An equivariant decoder processes the encoded context graph, and the latent coordinates of addresses
to produce a meaningful prediction at each desired hyper-edge.

Its generic formulation is the following:

.. math::
    \hat{y} = D_\theta^{equi}(x,h)

Multiple implementation are possible, but all should respect the following interface.

.. currentmodule:: energnn.gnn.decoder.equivariant_decoder

.. autoclass:: EquivariantDecoder
   :no-members:
   :show-inheritance:
   
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    EquivariantDecoder.init
    EquivariantDecoder.init_with_output
    EquivariantDecoder.apply


Implementations
===============

.. autoclass:: ZeroEquivariantDecoder
   :no-members:
   :show-inheritance:

.. autoclass:: MLPEquivariantDecoder
   :no-members:
   :show-inheritance: