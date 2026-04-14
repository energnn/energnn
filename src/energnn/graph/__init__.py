# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from .graph import (
    Graph,
    JaxGraph,
    check_hyper_edge_set_dict_type,
    collate_graphs,
    concatenate_graphs,
    get_statistics,
    separate_graphs,
)
from .hyper_edge_set import (
    HyperEdgeSet,
    JaxHyperEdgeSet,
    build_hyper_edge_set_shape,
    check_dict_or_none,
    check_dict_shape,
    check_no_nan,
    collate_hyper_edge_sets,
    concatenate_hyper_edge_sets,
    dict2array,
    dict2array as dict2array_jax,
    separate_hyper_edge_sets,
)
from .shape import (
    GraphShape,
    JaxGraphShape,
    collate_shapes,
    max_shape,
    separate_shapes,
    sum_shapes,
)
from .structure import GraphStructure, HyperEdgeSetStructure
from .utils import jnp_to_np, np_to_jnp, to_numpy

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
    "dict2array_jax",
    "JaxGraph",
    "JaxGraphShape",
    "np_to_jnp",
    "jnp_to_np",
]
