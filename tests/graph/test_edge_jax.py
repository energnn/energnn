#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from energnn.graph.edge import Edge
from energnn.graph.jax.edge import JaxEdge
from energnn.graph.jax.utils import np_to_jnp, jnp_to_np
from tests.graph.utils import get_fixed_edge, assert_edges_equal


def test_from_numpy_edge_and_to_numpy_edge_roundtrip():
    np_edge = get_fixed_edge()

    # Convert to JaxEdge
    jax_edge = JaxEdge.from_numpy_edge(np_edge, device=None, dtype="float32")
    # Check internals are JAX arrays / dicts
    assert isinstance(jax_edge.feature_array, jax.Array) or jax_edge.feature_array is None
    for v in jax_edge.address_dict.values():
        assert isinstance(v, jax.Array)

    # Convert back to numpy Edge and compare
    np_edge_round = jax_edge.to_numpy_edge()
    assert isinstance(np_edge_round, Edge)
    assert_edges_equal(np_edge, np_edge_round)


def test_from_numpy_edge_dtypes_64():
    jax.config.update("jax_enable_x64", True)
    try:
        np_edge = get_fixed_edge()
        jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float64")
        # feature_array should have dtype float64 in JAX
        assert jax_edge.feature_array.dtype == jnp.float64
        # and back to numpy: dtype preserved as float64
        back = jax_edge.to_numpy_edge()
        assert back.feature_array.dtype == np.float64
    finally:
        jax.config.update("jax_enable_x64", False)


def test_from_numpy_edge_dtypes_32():
    np_edge = get_fixed_edge()
    jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float32")
    # feature_array should have dtype float32 in JAX
    assert jax_edge.feature_array.dtype == jnp.float32
    # and back to numpy: dtype preserved as float32
    back = jax_edge.to_numpy_edge()
    assert back.feature_array.dtype == np.float32


def test_from_numpy_edge_dtypes_16():
    np_edge = get_fixed_edge()
    jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float16")
    # feature_array should have dtype float16 in JAX
    assert jax_edge.feature_array.dtype == jnp.float16
    # and back to numpy: dtype preserved as float16
    back = jax_edge.to_numpy_edge()
    assert back.feature_array.dtype == np.float16


def test_feature_flat_array_single_and_batch():
    # Single
    np_edge = get_fixed_edge()
    jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float32")
    # feature_array shape should be (n_obj, n_feats)
    assert len(jax_edge.feature_array.shape) == 2
    flat = jax_edge.feature_flat_array
    # single -> 1D
    assert flat.ndim == 1
    assert flat.shape[0] == np_edge.n_obj * jax_edge.feature_array.shape[-1]
    # Batch: stack two identical edges into a batch dimension
    jax_feat_batch = jnp.stack([jax_edge.feature_array, jax_edge.feature_array], axis=0)  # (2, n_obj, n_feats)
    jax_edge_batch = JaxEdge(
        address_dict=np_to_jnp(np_edge.address_dict),
        feature_array=jax_feat_batch,
        feature_names=np_to_jnp(np_edge.feature_names),
        non_fictitious=np_to_jnp(np_edge.non_fictitious),
    )
    flat_batch = jax_edge_batch.feature_flat_array
    assert flat_batch.ndim == 2
    assert flat_batch.shape[0] == 2
    assert flat_batch.shape[1] == np_edge.n_obj * jax_edge.feature_array.shape[-1]


def test_feature_flat_array_invalid_dims_raises():
    # Create a JaxEdge with invalid feature_array dims (1D)
    bad_feat = jnp.array([1.0, 2.0, 3.0])
    jax_edge = JaxEdge(
        address_dict=None,
        feature_array=bad_feat,
        feature_names={"a": jnp.array(0)},
        non_fictitious=jnp.array([1.0]),
    )
    with pytest.raises(ValueError):
        _ = jax_edge.feature_flat_array


def test_pytree_flatten_and_unflatten_roundtrip():
    np_edge = get_fixed_edge()
    jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float32")

    # Use JAX tree utilities to flatten and unflatten
    children, aux = jax.tree_util.tree_flatten(jax_edge)
    reconstructed = jax.tree_util.tree_unflatten(aux, children)
    # reconstructed is a JaxEdge; convert to numpy and compare to original numpy edge
    assert isinstance(reconstructed, JaxEdge)
    np_round = reconstructed.to_numpy_edge()
    assert_edges_equal(np_edge, np_round)


def test_tree_unflatten_classmethod_missing_keys_raises_keyerror():
    """
    Directly call the classmethod tree_unflatten with insufficient aux_data
    to trigger a KeyError inside (zipping will create a dict missing required keys).
    """
    # Prepare children matching the number of expected keys, but provide wrong aux_data
    np_edge = get_fixed_edge()
    jax_edge = JaxEdge.from_numpy_edge(np_edge, dtype="float32")
    children = list(jax_edge.values())
    # Provide aux_data missing required key strings
    aux_data = ("some", "keys", "not", "matching")
    with pytest.raises(KeyError):
        JaxEdge.tree_unflatten(aux_data, children)
