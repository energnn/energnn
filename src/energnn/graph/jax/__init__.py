# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# test

from .graph import (JaxGraph,
                    collate_graphs_jax,
                    concatenate_graphs_jax,
                    get_statistics_jax,
                    separate_graphs_jax,
                    check_hyper_edge_set_dict_type_jax)
from .hyper_edge_set import (JaxHyperEdgeSet,
                             collate_hyper_edge_sets_jax,
                             concatenate_hyper_edge_sets_jax,
                             separate_hyper_edge_sets_jax,
                             check_dict_shape_jax,
                             build_hyper_edge_set_shape_jax,
                             dict2array_jax,
                             check_no_nan_jax)
from .shape import (JaxGraphShape,
                    collate_shapes_jax,
                    max_shape_jax,
                    separate_shapes_jax,
                    sum_shapes_jax)
from .utils import jnp_to_np, np_to_jnp

__all__ = [
    "JaxHyperEdgeSet",
    "collate_hyper_edge_sets_jax",
    "concatenate_hyper_edge_sets_jax",
    "separate_hyper_edge_sets_jax",
    "check_dict_shape_jax",
    "build_hyper_edge_set_shape_jax",
    "dict2array_jax",
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
