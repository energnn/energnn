# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from .graph import Graph, check_hyper_edge_set_dict_type, collate_graphs, concatenate_graphs, get_statistics, separate_graphs
from .hyper_edge_set import (
    HyperEdgeSet,
    build_hyper_edge_set_shape,
    check_dict_or_none,
    check_dict_shape,
    check_no_nan,
    collate_hyper_edge_sets,
    concatenate_hyper_edge_sets,
    dict2array,
    separate_hyper_edge_sets,
)
from .jax.graph import (
    JaxGraph,
    collate_graphs_jax,
    concatenate_graphs_jax,
    get_statistics_jax,
    separate_graphs_jax,
    check_hyper_edge_set_dict_type_jax,
)
from .jax.hyper_edge_set import (
    JaxHyperEdgeSet,
    collate_hyper_edge_sets_jax,
    concatenate_hyper_edge_sets_jax,
    separate_hyper_edge_sets_jax,
    check_dict_shape_jax,
    build_hyper_edge_set_shape_jax,
    dict2array_jax,
    check_dict_or_none_jax,
    check_no_nan_jax,
)
from .jax.shape import JaxGraphShape, collate_shapes_jax, max_shape_jax, separate_shapes_jax, sum_shapes_jax
from .jax.utils import jnp_to_np, np_to_jnp
from .shape import GraphShape, collate_shapes, max_shape, separate_shapes, sum_shapes
from .structure import GraphStructure, HyperEdgeSetStructure
from .utils import to_numpy

__all__ = [
    "HyperEdgeSet",
    "collate_hyper_edge_sets",
    "concatenate_hyper_edge_sets",
    "separate_hyper_edge_sets",
    "check_dict_shape",
    "build_hyper_edge_set_shape",
    "dict2array",
    "check_dict_or_none",
    "check_no_nan",
    "Graph",
    "collate_graphs",
    "concatenate_graphs",
    "get_statistics",
    "separate_graphs",
    "check_hyper_edge_set_dict_type",
    "GraphShape",
    "collate_shapes",
    "GraphStructure",
    "HyperEdgeSetStructure",
    "max_shape",
    "separate_shapes",
    "sum_shapes",
    "to_numpy",
    "JaxHyperEdgeSet",
    "collate_hyper_edge_sets_jax",
    "concatenate_hyper_edge_sets_jax",
    "separate_hyper_edge_sets_jax",
    "check_dict_shape_jax",
    "build_hyper_edge_set_shape_jax",
    "dict2array_jax",
    "check_dict_or_none_jax",
    "check_no_nan_jax",
    "JaxGraph",
    "collate_graphs_jax",
    "concatenate_graphs_jax",
    "get_statistics_jax",
    "separate_graphs_jax",
    "check_hyper_edge_set_dict_type_jax",
    "JaxGraphShape",
    "collate_shapes_jax",
    "max_shape_jax",
    "separate_shapes_jax",
    "sum_shapes_jax",
    "np_to_jnp",
    "jnp_to_np",
]
