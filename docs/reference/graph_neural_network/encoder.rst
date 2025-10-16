========================
Encoder :math:`E_\theta`
========================

The encoder is a permutation-equivariant mapping that embeds input features into an abstract latent space.
It outputs a structure equivalent to the context graph, where numerical features have been modified.
Its generic formulation is as follows,

.. math::
    \tilde{x} = E_\theta(x).

Multiple implementation are possible, but all should respect the following interface.

.. currentmodule:: energnn.gnn

.. autoclass:: Encoder
   :no-members:
   :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

        Encoder.init
        Encoder.init_with_output
        Encoder.apply


Implementations
----------------
.. autoclass:: IdentityEncoder
   :no-members:
   :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

        IdentityEncoder.init
        IdentityEncoder.init_with_output
        IdentityEncoder.apply

.. autoclass:: MLPEncoder
   :no-members:
   :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :nosignatures:

        MLPEncoder.__call__
