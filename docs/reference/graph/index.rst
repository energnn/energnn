=====
Graph
=====

.. currentmodule:: energnn.graph

In this package, the classes :class:`Graph` and :class:`JaxGraph` are the core data representation.
There are used to represent contexts :math:`x` (*i.e* input data),
decisions :math:`y` (*i.e.* output data), and gradients :math:`\nabla_y f`.
A :class:`Graph` (resp :class:`JaxGraph`)is composed of multiple classes of hyper-edges
:class:`Edge` (resp :class:`JaxEdge`), each defined by a series of addresses and features.

The class :class:`Graph` or :class:`JaxGraph` can represent both a single graph instance or a batch of
graphs. 


.. note::
    :class:`JaxGraph` (resp :class:`JaxEdge`, resp :class:`JaxGraphShape`) is the Jax implementation
    of :class:`Graph` (resp :class:`Edge`, resp :class:`GraphShape`) which is based on numpy.
    Here is a typical instance of :class:`Graph` or :class:`JaxGraph`.

    .. code:: python

        >>> print(graph)
        Mass
                  addresses  features
                    node_id    weight         x         y         z
        object_id
        0               0.0  5.322265  0.202435  0.202435  0.242032
        1               1.0  3.496568  0.962326  0.962326  0.306690
        2               2.0  3.535864  0.060886  0.060886  0.094170
        3               3.0  7.213709  0.984766  0.984766  0.068853
        Spring
                  addresses           features
                   node1_id node2_id         k
        object_id
        0               0.0      1.0  0.020424
        1               1.0      2.0  0.037591
        2               2.0      3.0  0.045405
        Registry
        [0. 1. 2. 3.]


Graph
=====

.. autoclass:: Graph
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    Graph.from_dict
    Graph.to_pickle
    Graph.from_pickle
    Graph.is_batch
    Graph.is_single
    Graph.feature_flat_array
    Graph.pad
    Graph.unpad
    Graph.count_connected_components
    Graph.offset_addresses
    Graph.quantiles


.. autoclass:: JaxGraph
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    JaxGraph.tree_flatten
    JaxGraph.tree_unflatten
    JaxGraph.feature_flat_array
    JaxGraph.from_numpy_graph
    JaxGraph.to_numpy_graph
    JaxGraph.quantiles

Edge
====

.. autoclass:: Edge
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    Edge.from_dict
    Edge.array
    Edge.is_batch
    Edge.is_single
    Edge.n_obj
    Edge.n_batch
    Edge.address_array
    Edge.address_names
    Edge.feature_dict
    Edge.feature_flat_array
    Edge.pad
    Edge.unpad
    Edge.offset_addresses


.. autoclass:: JaxEdge
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    JaxEdge.tree_flatten
    JaxEdge.tree_unflatten
    JaxEdge.feature_flat_array
    JaxEdge.from_numpy_edge
    JaxEdge.to_numpy_edge


GraphShape
==========

.. autoclass:: GraphShape
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    GraphShape.from_dict
    GraphShape.to_jsonable_dict
    GraphShape.from_jsonable_dict
    GraphShape.max
    GraphShape.sum
    GraphShape.array
    GraphShape.is_single
    GraphShape.is_batch
    GraphShape.n_batch


.. autoclass:: JaxGraphShape
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    JaxGraphShape.tree_flatten
    JaxGraphShape.tree_unflatten
    JaxGraphShape.from_numpy_shape
    JaxGraphShape.to_numpy_shape


Graph, edge, and shape manipulation functions
=============================================
The following functions help to manipulate graphs, edges, shapes objects and to proceed oprations on them.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    collate_graphs
    concatenate_graphs
    get_statistics
    separate_graphs
    check_edge_dict_type
    collate_edges
    concatenate_edges
    separate_edges
    check_dict_shape
    build_edge_shape
    dict2array
    check_dict_or_none
    check_no_nan
    collate_shapes
    max_shape
    separate_shapes
    sum_shapes
    to_numpy
    np_to_jnp
    jnp_to_np