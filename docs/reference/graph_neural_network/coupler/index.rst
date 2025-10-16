========================
Coupler :math:`C_\theta`
========================

The coupler is a permutation-equivariant mapping that associates an encoded context :math:`x`
with coordinates for each of its addresses.
Given an encoded context graph :math:`x`, we decompose this mapping as follows:

.. math::
    C_\theta(x) = S(F_\theta(\cdot; x)).

The endomorphism :math:`F_\theta` is denoted as a *coupling function*, while
the functional :math:`S` is denoted as the *solving method*.


.. currentmodule:: energnn.gnn.coupler

.. autoclass:: Coupler
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
    Coupler.init
    Coupler.init_with_output
    Coupler.apply


Components
==========

.. toctree::
    :maxdepth: 1

    coupling_function/index
    solving_method
