============================================================
Self Message Function :math:`\psi^{\circlearrowleft}_\theta`
============================================================

The self message function processes the latent coordinates of an address :math:`a` as follows:

.. math::
    h_a^\circlearrowleft = \psi_\theta^\circlearrowleft(h_a).

Multiple implementations are possible, but all should respect the following interface.

.. currentmodule:: energnn.gnn.coupler.coupling_function.self_message_function

.. autoclass:: SelfMessageFunction
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    SelfMessageFunction.init
    SelfMessageFunction.init_with_output
    SelfMessageFunction.apply

Implementations
===============

.. autoclass:: EmptySelfMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: IdentitySelfMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: MLPSelfMessageFunction
   :no-members:
   :show-inheritance:
