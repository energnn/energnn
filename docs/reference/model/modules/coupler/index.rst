=======
Coupler
=======

Dire un truc sur le rôle général du coupler.
Dire que ça fait souvent intervenir des petites fonction de message, qui doivent respecter aussi une interface
et dont certaines implémentations sont fournies en XXX.


.. currentmodule:: energnn.model.coupler

.. autoclass:: Coupler
   :no-members:
   :show-inheritance:

Implementations
---------------

.. autoclass:: RecurrentCoupler
   :no-members:
   :show-inheritance:

.. autoclass:: NeuralODECoupler
   :no-members:
   :show-inheritance:

Message Functions
-----------------

.. autoclass:: MessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: IdentityMessageFunction
   :no-members:
   :show-inheritance:

.. autoclass:: LocalSumMessageFunction
   :no-members:
   :show-inheritance:
