# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.jax.hyper_edge_set import (
    JaxHyperEdgeSet,
    build_hyper_edge_set_shape_jax,
    check_dict_shape_jax,
    check_dict_or_none_jax,
    check_no_nan_jax,
    check_valid_ports_jax,
    collate_hyper_edge_sets_jax,
    concatenate_hyper_edge_sets_jax,
    dict2array_jax,
    separate_hyper_edge_sets_jax,
)
from energnn.graph.jax.utils import np_to_jnp
from tests.graph.utils import assert_edges_equal, get_fixed_edge
from tests.graph.utils_jax import get_fixed_edge_jax


def test_from_numpy_edge_and_to_numpy_edge_roundtrip():
    np_edge = get_fixed_edge()

    # Convert to JaxEdge
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, device=None, dtype="float32")
    # Check internals are JAX arrays / dicts
    assert isinstance(jax_edge.feature_array, jax.Array) or jax_edge.feature_array is None
    for v in jax_edge.port_dict.values():
        assert isinstance(v, jax.Array)

    # Convert back to numpy Edge and compare
    np_edge_round = jax_edge.to_numpy_hyper_edge_set()
    assert isinstance(np_edge_round, HyperEdgeSet)
    assert_edges_equal(np_edge, np_edge_round)


def test_from_numpy_edge_dtypes_64():
    jax.config.update("jax_enable_x64", True)
    try:
        np_edge = get_fixed_edge()
        jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float64")
        # feature_array should have dtype float64 in JAX
        assert jax_edge.feature_array.dtype == jnp.float64
        # and back to numpy: dtype preserved as float64
        back = jax_edge.to_numpy_hyper_edge_set()
        assert back.feature_array.dtype == np.float64
    finally:
        jax.config.update("jax_enable_x64", False)


def test_from_numpy_edge_dtypes_32():
    np_edge = get_fixed_edge()
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float32")
    # feature_array should have dtype float32 in JAX
    assert jax_edge.feature_array.dtype == jnp.float32
    # and back to numpy: dtype preserved as float32
    back = jax_edge.to_numpy_hyper_edge_set()
    assert back.feature_array.dtype == np.float32


def test_from_numpy_edge_dtypes_16():
    np_edge = get_fixed_edge()
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float16")
    # feature_array should have dtype float16 in JAX
    assert jax_edge.feature_array.dtype == jnp.float16
    # and back to numpy: dtype preserved as float16
    back = jax_edge.to_numpy_hyper_edge_set()
    assert back.feature_array.dtype == np.float16


def test_feature_flat_array_single_and_batch():
    # Single
    np_edge = get_fixed_edge()
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float32")
    # feature_array shape should be (n_obj, n_feats)
    assert len(jax_edge.feature_array.shape) == 2
    flat = jax_edge.feature_flat_array
    # single -> 1D
    assert flat.ndim == 1
    assert flat.shape[0] == np_edge.n_obj * jax_edge.feature_array.shape[-1]
    # Batch: stack two identical edges into a batch dimension
    jax_feat_batch = jnp.stack([jax_edge.feature_array, jax_edge.feature_array], axis=0)  # (2, n_obj, n_feats)
    jax_edge_batch = JaxHyperEdgeSet(
        port_dict=np_to_jnp(np_edge.port_dict),
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
    jax_edge = JaxHyperEdgeSet(
        port_dict=None,
        feature_array=bad_feat,
        feature_names={"a": jnp.array(0)},
        non_fictitious=jnp.array([1.0]),
    )
    with pytest.raises(ValueError):
        _ = jax_edge.feature_flat_array


def test_pytree_flatten_and_unflatten_roundtrip():
    np_edge = get_fixed_edge()
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float32")

    # Use JAX tree utilities to flatten and unflatten
    children, aux = jax.tree_util.tree_flatten(jax_edge)
    reconstructed = jax.tree_util.tree_unflatten(aux, children)
    # reconstructed is a JaxEdge; convert to numpy and compare to original numpy edge
    assert isinstance(reconstructed, JaxHyperEdgeSet)
    np_round = reconstructed.to_numpy_hyper_edge_set()
    assert_edges_equal(np_edge, np_round)


def test_tree_unflatten_classmethod_missing_keys_raises_keyerror():
    """
    Directly call the classmethod tree_unflatten with insufficient aux_data
    to trigger a KeyError inside (zipping will create a dict missing required keys).
    """
    # Prepare children matching the number of expected keys, but provide wrong aux_data
    np_edge = get_fixed_edge()
    jax_edge = JaxHyperEdgeSet.from_numpy_hyper_edge_set(np_edge, dtype="float32")
    children = list(jax_edge.values())
    # Provide aux_data missing required key strings
    aux_data = ("some", "keys", "not", "matching")
    with pytest.raises(KeyError):
        JaxHyperEdgeSet.tree_unflatten(aux_data, children)


def test_from_dict_and_basic_props():
    edge = get_fixed_edge_jax()
    # array concatenation shape: (n_obj, n_feats + n_ports) -> (2, 2 + 2) = (2,4)
    assert edge.array.shape == (2, 4)
    assert edge.is_single is True
    assert edge.is_batch is False
    assert edge.n_obj == 2
    # feature_dict returns same arrays
    fd = edge.feature_dict
    assert "b" in fd and "w" in fd
    chex.assert_trees_all_close(fd["b"], jnp.array([0.1, 0.2], dtype=jnp.float32))
    chex.assert_trees_all_close(fd["w"], jnp.array([0.5, 1.0], dtype=jnp.float32))


def test_feature_flat_array_getter_and_setter():
    edge = get_fixed_edge_jax()
    flat = edge.feature_flat_array
    # For single: 1D length n_obj * n_feats = 2 * 2 = 4
    assert flat.shape == (4,)
    # Create new flat array with same shape, set it, and verify feature_array changed accordingly
    new_flat = flat + 1.0
    edge.feature_flat_array = new_flat
    # reshape Fortran order: first column then second column -> check sum changed
    chex.assert_trees_all_close(edge.feature_array, new_flat.reshape([edge.n_obj, -1], order="F"))
    # the setter should not raise when shapes match; check final flat matches
    chex.assert_trees_all_close(edge.feature_flat_array, new_flat)


def test_feature_flat_array_setter_shape_mismatch_raises():
    edge = get_fixed_edge_jax()
    flat = edge.feature_flat_array
    with pytest.raises(ValueError):
        # wrong shape
        edge.feature_flat_array = flat[:-1]


def test_pad_and_unpad_single():
    edge = get_fixed_edge_jax()
    old_n = edge.n_obj
    edge.pad(4)
    assert edge.n_obj == 4
    # feature padding should add rows (zeros)
    assert edge.feature_array.shape[0] == 4
    # addresses padded too
    for k, v in edge.port_dict.items():
        assert v.shape[0] == 4
    # unpad back to original
    edge.unpad(old_n)
    assert edge.n_obj == old_n
    assert edge.feature_array.shape[0] == old_n


def test_pad_on_batch_raises():
    e1 = get_fixed_edge_jax()
    e2 = get_fixed_edge_jax()
    batch = collate_hyper_edge_sets_jax([e1, e2])
    with pytest.raises(ValueError):
        batch.pad(10)


def test_unpad_on_batch_raises():
    e1 = get_fixed_edge_jax()
    e2 = get_fixed_edge_jax()
    batch = collate_hyper_edge_sets_jax([e1, e2])
    with pytest.raises(ValueError):
        batch.unpad(1)


def test_offset_addresses():
    edge = get_fixed_edge_jax()
    orig = {k: v.copy() for k, v in edge.port_dict.items()}
    edge.offset_addresses(10)
    for k in orig:
        chex.assert_trees_all_close(edge.port_dict[k], orig[k] + 10)


def test_collate_and_separate_roundtrip():
    e1 = get_fixed_edge_jax()
    # modify e2 slightly so we can check separation
    e2 = get_fixed_edge_jax()
    e2.offset_addresses(100)
    batch = collate_hyper_edge_sets_jax([e1, e2])
    assert batch.is_batch is True
    assert batch.n_batch == 2
    separated = separate_hyper_edge_sets_jax(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # first separated edge should equal e1 (addresses, features)
    chex.assert_trees_all_equal(separated[0].port_dict["dst"], e1.port_dict["dst"])
    chex.assert_trees_all_equal(separated[1].port_dict["dst"], e2.port_dict["dst"])


def test_collate_empty_raises():
    with pytest.raises(IndexError):
        collate_hyper_edge_sets_jax([])


def test_collate_inconsistent_keys_raises():
    e1 = get_fixed_edge_jax()
    e2 = get_fixed_edge_jax()
    # remove addresses from e2 to produce mismatch
    e2.port_dict = None
    with pytest.raises(ValueError):
        collate_hyper_edge_sets_jax([e1, e2])


def test_concatenate_edges():
    e1 = get_fixed_edge_jax()
    e2 = get_fixed_edge_jax()
    concatenated = concatenate_hyper_edge_sets_jax([e1, e2])
    # n_obj should be sum
    assert concatenated.n_obj == e1.n_obj + e2.n_obj
    # addresses concatenated
    chex.assert_trees_all_equal(concatenated.port_dict["dst"][:2], e1.port_dict["dst"])
    chex.assert_trees_all_equal(concatenated.port_dict["dst"][2:], e2.port_dict["dst"])


def test_check_dict_shape_and_build_edge_shape_errors():
    # mismatching last dims
    d1 = {"a": jnp.zeros((3,), dtype=jnp.float32), "b": jnp.zeros((4,), dtype=jnp.float32)}
    with pytest.raises(ValueError):
        check_dict_shape_jax(d=d1, n_objects=None)
    # build_edge_shape with both None should raise
    with pytest.raises(ValueError):
        build_hyper_edge_set_shape_jax(port_dict=None, feature_dict=None)


def test_dict2array_and_sorting():
    d = {"z": jnp.array([1, 2], dtype=jnp.float32), "a": jnp.array([3, 4], dtype=jnp.float32)}
    arr = dict2array_jax(d)
    # keys sorted -> ['a', 'z'] -> columns correspond
    chex.assert_trees_all_equal(arr[:, 0], d["a"])
    chex.assert_trees_all_equal(arr[:, 1], d["z"])
    assert dict2array_jax(None) is None


def test_check_dict_or_none_and_nan_and_valid_addresses():
    # non-dict non-None raises
    with pytest.raises(ValueError):
        check_dict_or_none_jax(jnp.array([1, 2, 3]))
    # NaN detection in address
    with pytest.raises(ValueError):
        check_no_nan_jax(port_dict={"a": jnp.array([0.0, jnp.nan], dtype=jnp.float32)}, feature_dict=None)
    # NaN detection in feature
    with pytest.raises(ValueError):
        check_no_nan_jax(port_dict=None, feature_dict={"f": jnp.array([jnp.nan, 1.0], dtype=jnp.float32)})
    # valid addresses: float values but integer-valued pass (1.0)
    check_valid_ports_jax({"x": jnp.array([1.0, 2.0], dtype=jnp.float32)})
    # non-integer valued should raise
    with pytest.raises(ValueError):
        check_valid_ports_jax({"x": jnp.array([1.0, 2.3], dtype=jnp.float32)})


def test_str_repr_single_and_batch():
    e = get_fixed_edge_jax()
    s = str(e)
    assert "features" in s and "ports" in s
    # batch
    b = collate_hyper_edge_sets_jax([e, e])
    s2 = str(b)
    assert "features" in s2 and "ports" in s2
