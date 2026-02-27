=======
Model
=======

.. currentmodule:: energnn.model

Our core GNN implementations have been designed to process H2MG data.
There are 3 ways of constructing a GNN.

- Build a :class:`~energnn.model.SimpleGNN` by combining the various module implementations provided in **EnerGNN**.
- Build a :class:`~energnn.model.SimpleGNN` by combining your own module implementations with the ones
  that are pre-existing in **EnerGNN** (just make sure to respect interfaces).
- Use a **ready-to-use** GNN implementation provided in the :mod:`energnn.model.ready_to_use` module.

Components
----------

.. toctree::
    :maxdepth: 1

    core_gnn/index
    modules/index
    ready_to_use/index
