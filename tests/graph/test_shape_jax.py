#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph.jax.shape import JaxGraphShape, collate_shapes_jax, max_shape_jax, separate_shapes_jax, sum_shapes_jax
from energnn.graph.shape import GraphShape
from tests.graph.utils import assert_graphshape_equal, get_fixed_graphshape
from tests.graph.utils_jax import get_fixed_edge_jax


def test_from_numpy_and_to_numpy_roundtrip():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")
    # internals are jax arrays
    for v in jgs.hyper_edge_sets.values():
        assert isinstance(v, jax.Array)
    assert isinstance(jgs.addresses, jax.Array)

    back = jgs.to_numpy_shape()
    assert isinstance(back, GraphShape)
    assert_graphshape_equal(gs, back)


def test_dtype_preservation_float64():
    jax.config.update("jax_enable_x64", True)
    try:
        gs = get_fixed_graphshape()
        jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float64")
        # jax arrays should be float64
        for v in jgs.hyper_edge_sets.values():
            assert v.dtype == jnp.float64
        assert jgs.addresses.dtype == jnp.float64

        back = jgs.to_numpy_shape()
        for v in back.hyper_edge_sets.values():
            assert v.dtype == np.float64
        assert back.addresses.dtype == np.float64
    finally:
        jax.config.update("jax_enable_x64", False)


def test_dtype_preservation_float32():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")
    # jax arrays should be float32
    for v in jgs.hyper_edge_sets.values():
        assert v.dtype == jnp.float32
    assert jgs.addresses.dtype == jnp.float32

    back = jgs.to_numpy_shape()
    for v in back.hyper_edge_sets.values():
        assert v.dtype == np.float32
    assert back.addresses.dtype == np.float32


def test_dtype_preservation_float16():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float16")
    # jax arrays should be float16
    for v in jgs.hyper_edge_sets.values():
        assert v.dtype == jnp.float16
    assert jgs.addresses.dtype == jnp.float16

    back = jgs.to_numpy_shape()
    for v in back.hyper_edge_sets.values():
        assert v.dtype == np.float16
    assert back.addresses.dtype == np.float16


def test_pytree_flatten_unflatten_roundtrip():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")

    children, aux = jax.tree_util.tree_flatten(jgs)
    reconstructed = jax.tree_util.tree_unflatten(aux, children)
    assert isinstance(reconstructed, JaxGraphShape)

    # convert back to numpy and compare
    back = reconstructed.to_numpy_shape()
    assert_graphshape_equal(gs, back)


def test_tree_unflatten_classmethod_missing_keys_raises_keyerror():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")
    children, aux = jgs.tree_flatten()
    # Provide aux_data missing required elements
    aux_data = ("bad",) # Only one element
    with pytest.raises(ValueError):
        JaxGraphShape.tree_unflatten(aux_data, children)


def test_from_dict_and_to_json_roundtrip():
    e = get_fixed_edge_jax()
    # non_fictitious array
    non_fictitious = jnp.ones((5,), dtype=jnp.float32)
    gs = JaxGraphShape.from_dict(hyper_edge_set_dict={"edge_type": e}, non_fictitious=non_fictitious)

    # to jsonable and back
    js = gs.to_jsonable_dict()
    assert "hyper_edge_sets" in js and "addresses" in js
    gs2 = JaxGraphShape.from_jsonable_dict(js)
    # edges keys equal and addresses equal
    assert set(gs.hyper_edge_sets.keys()) == set(gs2.hyper_edge_sets.keys())
    assert int(gs.addresses) == int(gs2.addresses)


def test_array_and_is_single_is_batch_and_n_batch():
    e = get_fixed_edge_jax()
    gs = JaxGraphShape.from_dict(hyper_edge_set_dict={"a": e, "b": e}, non_fictitious=jnp.ones((3,)))
    arr = gs.array
    # since edges.values are scalars -> array 1-D with length 2 (two edge classes)
    assert arr.ndim == 1
    assert gs.is_single is True
    assert gs.is_batch is False
    with pytest.raises(ValueError):
        _ = gs.n_batch

    # Build a batched GraphShape manually: edges values should be arrays of shape (batch,)
    s1 = JaxGraphShape(hyper_edge_sets={"a": jnp.array(1), "b": jnp.array(2)}, addresses=jnp.array(0))
    s2 = JaxGraphShape(hyper_edge_sets={"a": jnp.array(3), "b": jnp.array(4)}, addresses=jnp.array(7))
    batched = collate_shapes_jax([s1, s2])
    assert batched.is_batch is True
    assert batched.n_batch == 2
    # array should now be 2-D (batch, n_edge_classes)
    assert batched.array.ndim == 2


def test_collate_shapes_empty_raises():
    with pytest.raises(ValueError):
        collate_shapes_jax([])


def test_separate_shapes_non_batched_raises():
    e = get_fixed_edge_jax()
    gs = JaxGraphShape.from_dict(hyper_edge_set_dict={"edge": e}, non_fictitious=jnp.ones((2,)))
    # gs is single -> separate should raise
    with pytest.raises(ValueError):
        separate_shapes_jax(gs)


def test_collate_and_separate_roundtrip():
    # build two JaxGraphShape with same keys
    e = get_fixed_edge_jax()
    gs1 = JaxGraphShape.from_dict(hyper_edge_set_dict={"t": e, "u": e}, non_fictitious=jnp.ones((2,)))
    gs2 = JaxGraphShape.from_dict(hyper_edge_set_dict={"t": e, "u": e}, non_fictitious=jnp.ones((4,)))
    batched = collate_shapes_jax([gs1, gs2])
    # separate back
    separated = separate_shapes_jax(batched)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # each item is a JaxGraphShape with same keys
    assert set(separated[0].hyper_edge_sets.keys()) == set(batched.hyper_edge_sets.keys())
    # addresses recovered
    assert separated[0].addresses.shape == ()
    assert separated[1].addresses.shape == ()


def test_max_and_sum_binary_operations():
    e = get_fixed_edge_jax()
    gs1 = JaxGraphShape.from_dict(hyper_edge_set_dict={"A": e, "B": e}, non_fictitious=jnp.ones((3,)))
    gs2 = JaxGraphShape(hyper_edge_sets={"A": jnp.array(5), "B": jnp.array(1)}, addresses=jnp.array(10))

    m = JaxGraphShape.max(gs1, gs2)
    # per-class maxima
    assert int(m.hyper_edge_sets["A"]) == max(int(gs1.hyper_edge_sets["A"]), int(gs2.hyper_edge_sets["A"]))
    assert int(m.addresses) == max(int(gs1.addresses), int(gs2.addresses))

    s = JaxGraphShape.sum(gs1, gs2)
    assert int(s.hyper_edge_sets["A"]) == int(gs1.hyper_edge_sets["A"]) + int(gs2.hyper_edge_sets["A"])
    assert int(s.addresses) == int(gs1.addresses) + int(gs2.addresses)


def test_max_shape_and_sum_shapes_list_ops_and_errors():
    e = get_fixed_edge_jax()
    gs1 = JaxGraphShape.from_dict(hyper_edge_set_dict={"X": e}, non_fictitious=jnp.ones((2,)))
    gs2 = JaxGraphShape(hyper_edge_sets={"X": jnp.array(5)}, addresses=jnp.array(10))

    # normal usage
    max_res = max_shape_jax([gs1, gs2])
    assert isinstance(max_res, JaxGraphShape)
    assert int(max_res.hyper_edge_sets["X"]) == max(int(gs1.hyper_edge_sets["X"]), int(gs2.hyper_edge_sets["X"]))

    sum_res = sum_shapes_jax([gs1, gs2])
    assert int(sum_res.hyper_edge_sets["X"]) == int(gs1.hyper_edge_sets["X"]) + int(gs2.hyper_edge_sets["X"])

    # errors on empty list
    with pytest.raises(ValueError):
        max_shape_jax([])

    with pytest.raises(ValueError):
        sum_shapes_jax([])

    # error if list contains non-GraphShape
    with pytest.raises(ValueError):
        max_shape_jax([gs1, "not a graphshape"])


def test_array_values_and_ordering_consistency():
    # Ensure that array stacks edges in whatever dict insertion order is used (we rely on .values())
    gs = JaxGraphShape(hyper_edge_sets={"first": jnp.array(1), "second": jnp.array(2)}, addresses=jnp.array(0))
    arr = gs.array
    # array should contain both values and length == number of edge classes
    assert arr.shape[-1] == 2
    assert set(gs.hyper_edge_sets.keys()) == {"first", "second"}
