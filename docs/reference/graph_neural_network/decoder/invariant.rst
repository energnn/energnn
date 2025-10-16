========================================
Invariant Decoder :math:`D^{inv}_\theta`
========================================


An equivariant decoder processes the encoded context graph, and the latent coordinates of addresses
to produce a global decision vector.

Its generic formulation is the following:

.. math::
    \hat{y} = D_\theta^{inv}(x,h).

Multiple implementation are possible, but all should respect the following interface.

.. currentmodule:: energnn.gnn.decoder.invariant_decoder

.. autoclass:: InvariantDecoder
   :no-members:
   :show-inheritance:
   
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    InvariantDecoder.init_with_size
    InvariantDecoder.init
    InvariantDecoder.init_with_output
    InvariantDecoder.apply


Implementations
===============

.. autoclass:: ZeroInvariantDecoder
   :no-members:
   :show-inheritance:

.. autoclass:: SumInvariantDecoder
   :no-members:
   :show-inheritance:

.. autoclass:: MeanInvariantDecoder
   :no-members:
   :show-inheritance:

.. autoclass:: AttentionInvariantDecoder
   :no-members:
   :show-inheritance:
