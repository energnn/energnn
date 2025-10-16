========================================================
Local Message Function :math:`\psi^{\rightarrow}_\theta`
========================================================

The local message function allows for addresses to share information with their
direct neighbors *w.r.t.* the context graph :math:`x`.
Its abstract formulation for an address :math:`a` is as follows:

.. math::
    h_a^\rightarrow = \psi_\theta^\rightarrow(h_a, \{ (h_e, x_e) \}_{(c,e,o)\in \mathcal{N}_{a}(x)}),

where :math:`\mathcal{N}_{a}(x)` is the unordered neighborhood of address :math:`a`.
The unordered set :math:`\{ (h_e, x_e) \}_{(c,e,o)\in \mathcal{N}_{a}(x)}` should
be aggregated in a permutation-invariant way, to preserve the permutation-equivariance
of the overall architecture.

Multiple implementations are possible, but all should respect the following interface.

.. currentmodule:: energnn.gnn.coupler.coupling_function.local_message_function
.. autoclass:: LocalMessageFunction
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    LocalMessageFunction.init
    LocalMessageFunction.init_with_output
    LocalMessageFunction.apply


Implementations
===============

.. autoclass:: EmptyLocalMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: IdentityLocalMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: SumLocalMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: AttentionLocalMessageFunction
   :no-members:
   :show-inheritance:

..
    Legacy
    ======

    .. autoclass:: GATv2LocalMessageFunction()
