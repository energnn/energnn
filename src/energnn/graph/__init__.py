# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from .edge import (
    Edge,
    build_edge_shape,
    check_dict_or_none,
    check_dict_shape,
    check_no_nan,
    collate_edges,
    concatenate_edges,
    dict2array,
    separate_edges,
)
from .graph import Graph, check_edge_dict_type, collate_graphs, concatenate_graphs, get_statistics, separate_graphs
from .jax.edge import JaxEdge
from .jax.graph import JaxGraph
from .jax.shape import JaxGraphShape
from .jax.utils import jnp_to_np, np_to_jnp
from .shape import GraphShape, collate_shapes, max_shape, separate_shapes, sum_shapes
from .structure import EdgeStructure, GraphStructure
from .utils import to_numpy

__all__ = [
    "Edge",
    "collate_edges",
    "concatenate_edges",
    "separate_edges",
    "check_dict_shape",
    "build_edge_shape",
    "dict2array",
    "check_dict_or_none",
    "check_no_nan",
    "Graph",
    "collate_graphs",
    "concatenate_graphs",
    "get_statistics",
    "separate_graphs",
    "check_edge_dict_type",
    "GraphShape",
    "collate_shapes",
    "max_shape",
    "separate_shapes",
    "sum_shapes",
    "to_numpy",
    "JaxEdge",
    "JaxGraphShape",
    "JaxGraph",
    "np_to_jnp",
    "jnp_to_np",
    "GraphStructure",
    "EdgeStructure",
]
