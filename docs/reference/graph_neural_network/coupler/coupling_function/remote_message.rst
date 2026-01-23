======================================================
Remote Message Function :math:`\psi^{\leadsto}_\theta`
======================================================

The remote message function allows for all addresses to share information remotely.
It abstract formulation for an address :math:`a` is as follows:

.. math::
    h_a^\leadsto = \psi_\theta^\leadsto(h_a, \{ h_{a'} \}_{a' \in \mathcal{A}(x)}).

The unordered set :math:`\{ h_{a'} \}_{a' \in \mathcal{A}(x)}` should
be aggregated in a permutation-invariant way, to preserve the permutation-equivariance
of the overall architecture.

Multiple implementations are possible, but all should respect the following interface.


.. currentmodule:: energnn.gnn.coupler.coupling_function.remote_message_function

.. autoclass:: RemoteMessageFunction
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    RemoteMessageFunction.init
    RemoteMessageFunction.init_with_output
    RemoteMessageFunction.apply

Implementations
_______________

.. autoclass:: EmptyRemoteMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: IdentityRemoteMessageFunction
   :no-members:
   :show-inheritance:
