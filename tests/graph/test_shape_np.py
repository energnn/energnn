#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import pytest

from energnn.graph.shape import (
    GraphShape,
    collate_shapes,
    separate_shapes,
    max_shape,
    sum_shapes,
)
from energnn.graph.edge import Edge
from tests.graph.utils import get_fixed_edge


def test_from_dict_and_to_json_roundtrip():
    e = get_fixed_edge()
    # non_fictitious array
    non_fictitious = np.ones((5,), dtype=np.float32)
    gs = GraphShape.from_dict(edge_dict={"edge_type": e}, non_fictitious=non_fictitious)

    # to jsonable and back
    js = gs.to_jsonable_dict()
    assert "edges" in js and "addresses" in js
    gs2 = GraphShape.from_jsonable_dict(js)
    # edges keys equal and addresses equal
    assert set(gs.edges.keys()) == set(gs2.edges.keys())
    assert int(gs.addresses) == int(gs2.addresses)


def test_array_and_is_single_is_batch_and_n_batch():
    e = get_fixed_edge()
    gs = GraphShape.from_dict(edge_dict={"a": e, "b": e}, non_fictitious=np.ones((3,)))
    arr = gs.array
    # since edges.values are scalars -> array 1-D with length 2 (two edge classes)
    assert arr.ndim == 1
    assert gs.is_single is True
    assert gs.is_batch is False
    with pytest.raises(ValueError):
        _ = gs.n_batch

    # Build a batched GraphShape manually: edges values should be arrays of shape (batch,)
    s1 = GraphShape(edges={"a": np.array(1), "b": np.array(2)}, addresses=np.array(0))
    s2 = GraphShape(edges={"a": np.array(3), "b": np.array(4)}, addresses=np.array(7))
    batched = collate_shapes([s1, s2])
    assert batched.is_batch is True
    assert batched.n_batch == 2
    # array should now be 2-D (batch, n_edge_classes)
    assert batched.array.ndim == 2


def test_collate_shapes_empty_raises():
    with pytest.raises(ValueError):
        collate_shapes([])


def test_separate_shapes_non_batched_raises():
    e = get_fixed_edge()
    gs = GraphShape.from_dict(edge_dict={"edge": e}, non_fictitious=np.ones((2,)))
    # gs is single -> separate should raise
    with pytest.raises(ValueError):
        separate_shapes(gs)


def test_collate_and_separate_roundtrip():
    # build two GraphShape with same keys
    e = get_fixed_edge()
    gs1 = GraphShape.from_dict(edge_dict={"t": e, "u": e}, non_fictitious=np.ones((2,)))
    gs2 = GraphShape.from_dict(edge_dict={"t": e, "u": e}, non_fictitious=np.ones((4,)))
    batched = collate_shapes([gs1, gs2])
    # separate back
    separated = separate_shapes(batched)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # each item is a GraphShape with same keys
    assert set(separated[0].edges.keys()) == set(batched.edges.keys())
    # addresses recovered
    assert separated[0].addresses.shape == ()
    assert separated[1].addresses.shape == ()


def test_max_and_sum_binary_operations():
    e = get_fixed_edge()
    gs1 = GraphShape.from_dict(edge_dict={"A": e, "B": e}, non_fictitious=np.ones((3,)))
    gs2 = GraphShape(edges={"A": np.array(5), "B": np.array(1)}, addresses=np.array(10))

    m = GraphShape.max(gs1, gs2)
    # per-class maxima
    assert int(m.edges["A"]) == max(int(gs1.edges["A"]), int(gs2.edges["A"]))
    assert int(m.addresses) == max(int(gs1.addresses), int(gs2.addresses))

    s = GraphShape.sum(gs1, gs2)
    assert int(s.edges["A"]) == int(gs1.edges["A"]) + int(gs2.edges["A"])
    assert int(s.addresses) == int(gs1.addresses) + int(gs2.addresses)


def test_max_shape_and_sum_shapes_list_ops_and_errors():
    e = get_fixed_edge()
    gs1 = GraphShape.from_dict(edge_dict={"X": e}, non_fictitious=np.ones((2,)))
    gs2 = GraphShape(edges={"X": np.array(5)}, addresses=np.array(10))

    # normal usage
    max_res = max_shape([gs1, gs2])
    assert isinstance(max_res, GraphShape)
    assert int(max_res.edges["X"]) == max(int(gs1.edges["X"]), int(gs2.edges["X"]))

    sum_res = sum_shapes([gs1, gs2])
    assert int(sum_res.edges["X"]) == int(gs1.edges["X"]) + int(gs2.edges["X"])

    # errors on empty list
    with pytest.raises(ValueError):
        max_shape([])

    with pytest.raises(ValueError):
        sum_shapes([])

    # error if list contains non-GraphShape
    with pytest.raises(ValueError):
        max_shape([gs1, "not a graphshape"])


def test_array_values_and_ordering_consistency():
    # Ensure that array stacks edges in whatever dict insertion order is used (we rely on .values())
    gs = GraphShape(edges={"first": np.array(1), "second": np.array(2)}, addresses=np.array(0))
    arr = gs.array
    # array should contain both values and length == number of edge classes
    assert arr.shape[-1] == 2
    assert set(gs.edges.keys()) == {"first", "second"}
