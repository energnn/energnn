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

from energnn.graph.shape import GraphShape
from energnn.graph.edge import Edge
from energnn.graph.jax.shape import JaxGraphShape
from tests.graph.utils import (
    get_fixed_graphshape,
    assert_graphshape_equal
)


def test_from_numpy_and_to_numpy_roundtrip():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")
    # internals are jax arrays
    for v in jgs.edges.values():
        assert isinstance(v, jax.Array)
    assert isinstance(jgs.addresses, jax.Array)

    back = jgs.to_numpy_shape()
    assert isinstance(back, GraphShape)
    assert_graphshape_equal(gs, back)


def test_dtype_preservation_float64():
    jax.config.update("jax_enable_x64", True)
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float64")
    # jax arrays should be float64
    for v in jgs.edges.values():
        assert v.dtype == jnp.float64
    assert jgs.addresses.dtype == jnp.float64

    back = jgs.to_numpy_shape()
    for v in back.edges.values():
        assert v.dtype == np.float64
    assert back.addresses.dtype == np.float64


def test_dtype_preservation_float32():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float32")
    # jax arrays should be float32
    for v in jgs.edges.values():
        assert v.dtype == jnp.float32
    assert jgs.addresses.dtype == jnp.float32

    back = jgs.to_numpy_shape()
    for v in back.edges.values():
        assert v.dtype == np.float32
    assert back.addresses.dtype == np.float32


def test_dtype_preservation_float16():
    gs = get_fixed_graphshape()
    jgs = JaxGraphShape.from_numpy_shape(gs, dtype="float16")
    # jax arrays should be float16
    for v in jgs.edges.values():
        assert v.dtype == jnp.float16
    assert jgs.addresses.dtype == jnp.float16

    back = jgs.to_numpy_shape()
    for v in back.edges.values():
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
    children = list(jgs.values())
    # wrong aux_data should raise KeyError
    aux_data = ("bad", "keys")
    with pytest.raises(KeyError):
        JaxGraphShape.tree_unflatten(aux_data, children)
