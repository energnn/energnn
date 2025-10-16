==================================
Coupling Function :math:`F_\theta`
==================================

A coupling function is an endomorphism of the space of address latent coordinates,
parameterized by the context graph :math:`x`.
For arbitrary address coordinates :math:`h`, it outputs coordinates :math:`h'` defined in the same space:

.. math::
    h' = F_\theta(h;x).

In this package, we choose to decompose the coupling function as:

.. math::
    \forall a\in \mathcal{A}(x), \ \ \ \ F_\theta(h;x)_a = \phi_\theta(h_a^\circlearrowleft, h_a^\rightarrow, h_a^\leadsto).

- :math:`\phi_\theta` is a Multi Layer Perceptron;

- :math:`h_a^\circlearrowleft` is the **self message**,
  that only depends on coordinates of address :math:`a`.
- :math:`h_a^\rightarrow`
  is the **local message**, that depends on the direct neighbors of :math:`a` *w.r.t.* the context graph :math:`x`.
- :math:`h_a^\leadsto`
  is the **remote message**, that depends on the coordinates of all addresses.

.. currentmodule:: energnn.gnn.coupler.coupling_function

.. autoclass:: CouplingFunction
  :no-members:
  :show-inheritance:

.. autosummary::
  :toctree: _autosummary
  :nosignatures:
  
    CouplingFunction.init
    CouplingFunction.init_with_output
    CouplingFunction.apply

Components
==========

.. toctree::
    :maxdepth: 1

    self_message
    local_message
    remote_message
