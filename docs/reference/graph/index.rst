=====
Graph
=====

.. currentmodule:: energnn.graph

In this package, the class :class:`Graph` is the core data representation.
It is used to represent contexts :math:`x` (*i.e* input data),
decisions :math:`y` (*i.e.* output data), and gradients :math:`\nabla_y f`.
A :class:`Graph` is composed of multiple :class:`HyperEdgeSet` objects,
each defined by a series of ports and features.

The class :class:`Graph` can represent both a single graph instance or a batch of
graphs. The backend (NumPy or JAX) is controlled by passing a :class:`NumpyBackend` or
:class:`JaxBackend` instance at construction time.

.. note::
    Here is a typical instance of :class:`Graph`.

    .. code:: python

        >>> print(graph)
        Mass
                  ports      features
                    node_id    weight         x         y         z
        object_id
        0               0.0  5.322265  0.202435  0.202435  0.242032
        1               1.0  3.496568  0.962326  0.962326  0.306690
        2               2.0  3.535864  0.060886  0.060886  0.094170
        3               3.0  7.213709  0.984766  0.984766  0.068853
        Spring
                  ports              features
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

HyperEdgeSet
============

.. autoclass:: HyperEdgeSet
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    HyperEdgeSet.from_dict
    HyperEdgeSet.array
    HyperEdgeSet.is_batch
    HyperEdgeSet.is_single
    HyperEdgeSet.n_obj
    HyperEdgeSet.n_batch
    HyperEdgeSet.port_array
    HyperEdgeSet.port_names
    HyperEdgeSet.feature_dict
    HyperEdgeSet.feature_flat_array
    HyperEdgeSet.pad
    HyperEdgeSet.unpad
    HyperEdgeSet.offset_addresses


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


Graph, hyper-edge set, and shape manipulation functions
=======================================================
The following functions help to manipulate graphs, hyper-edge sets, shapes objects and to proceed operations on them.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    collate_graphs
    concatenate_graphs
    get_statistics
    separate_graphs
    check_hyper_edge_set_dict_type
    collate_hyper_edge_sets
    concatenate_hyper_edge_sets
    separate_hyper_edge_sets
    check_dict_shape
    build_hyper_edge_set_shape
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