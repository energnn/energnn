# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import jax

from energnn.graph.jax import JaxEdge, JaxGraph
from energnn.normalizer.normalization_function import NormalizationFunction


def apply_tree(f: NormalizationFunction, params: dict, in_tree: dict, non_fictitious_tree: dict) -> dict:
    """
    Apply a normalization function over each edge in a graph tree structure.

    Maps the leaf-edge structures in `in_tree` and `non_fictitious_tree` through
    `f.apply`, using corresponding parameters from `params`.

    :param f: Instance of a NormalizationFunction to apply.
    :param params: Mapping from edge keys to normalization parameters.
    :param in_tree: Mapping from edge keys to JaxEdge input structures.
    :param non_fictitious_tree: Mapping from edge keys to masks indicating valid entries.
    :return: New edge tree with normalized feature arrays.
    """
    return jax.tree.map(f.apply, params, in_tree, non_fictitious_tree)


def apply_inverse_tree(f: NormalizationFunction, params: dict, in_tree: dict, non_fictitious_tree: dict) -> dict:
    """
    Apply the inverse normalization over each edge in a graph tree.

    Uses `f.apply_inverse` to revert normalized features to original scale.

    :param f: Normalization function instance.
    :param params: Mapping from edge keys to normalization parameters.
    :param in_tree: Input edge tree with normalized features.
    :param non_fictitious_tree: Mapping from edge keys to masks indicating valid entries.
    :return: Edge tree with denormalized feature arrays.
    """
    return jax.tree.map(f.apply_inverse, params, in_tree, non_fictitious_tree)


def gradient_inverse_tree(f: NormalizationFunction, params: dict, in_tree: dict, non_fictitious_tree: dict) -> dict:
    """
    Compute gradients of the inverse normalization for each edge in a graph tree.

    Applies `f.gradient_inverse` to obtain gradients of denormalization.

    :param f: Normalization function instance.
    :param params: Mapping from edge keys to normalization parameters.
    :param in_tree: Input edge tree with normalized features.
    :param non_fictitious_tree: Mapping from edge keys to masks indicating valid entries.
    :return: Tree mapping edge keys to gradient arrays.
    """
    return jax.tree.map(f.gradient_inverse, params, in_tree, non_fictitious_tree)


# def set_fictitious_to_zero(in_tree: dict) -> dict:
#     def zero(feature_array, non_fictitious):
#         return feature_array * jnp.expand_dims(non_fictitious, -1)
#
#     feature_tree = {k: edge.feature_array for k, edge in in_tree.items()}
#     non_fictitious_tree = {k: edge.non_fictitious for k, edge in in_tree.items()}
#     return jax.tree.map(zero, feature_tree, non_fictitious_tree)


def out_tree_to_graph(out_tree: dict, input_graph: JaxGraph) -> JaxGraph:
    """
    Construct a new JaxGraph by replacing feature arrays with output tree values.

    Retains original address mappings, feature names, and mask structure, but
    injects each output array from `out_tree` into the corresponding edge.

    :param out_tree: Mapping from edge keys to result feature arrays.
    :param input_graph: Original graph providing edge metadata.
    :return: New JaxGraph with updated feature arrays.
    """
    edges = {}
    for k, e in input_graph.edges.items():
        edges[k] = JaxEdge(
            address_dict=e.address_dict,
            feature_names=e.feature_names,
            feature_array=out_tree[k],
            non_fictitious=e.non_fictitious,
        )
    return JaxGraph(
        edges=edges,
        non_fictitious_addresses=input_graph.non_fictitious_addresses,
        true_shape=input_graph.true_shape,
        current_shape=input_graph.current_shape,
    )
