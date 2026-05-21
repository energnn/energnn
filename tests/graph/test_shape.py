# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.shape import GraphShape, collate_shapes, max_shape, separate_shapes, sum_shapes
from tests.graph.utils import get_fixed_edge


def test_from_dict_and_to_json_roundtrip(backend):
    e = get_fixed_edge(backend=backend)
    non_fictitious = np.ones(5, dtype=np.float32)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"edge_type": e}, non_fictitious=non_fictitious, backend=backend)
    js = gs.to_jsonable_dict()
    assert "hyper_edge_sets" in js and "addresses" in js
    gs2 = GraphShape.from_jsonable_dict(js, backend=backend)
    assert set(gs.hyper_edge_sets.keys()) == set(gs2.hyper_edge_sets.keys())
    assert int(gs.addresses) == int(gs2.addresses)


def test_backend_stored_correctly(backend):
    e = get_fixed_edge(backend=backend)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"e": e}, non_fictitious=np.ones(3), backend=backend)
    assert gs._backend == backend


def test_array_and_is_single_is_batch_and_n_batch(backend):
    e = get_fixed_edge(backend=backend)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"a": e, "b": e}, non_fictitious=np.ones(3), backend=backend)
    arr = gs.array
    assert arr.ndim == 1
    assert gs.is_single is True
    assert gs.is_batch is False
    with pytest.raises(ValueError):
        _ = gs.n_batch

    xp = backend.xp
    s1 = GraphShape(backend=backend, hyper_edge_sets={"a": xp.array(1), "b": xp.array(2)}, addresses=xp.array(0))
    s2 = GraphShape(backend=backend, hyper_edge_sets={"a": xp.array(3), "b": xp.array(4)}, addresses=xp.array(7))
    batched = collate_shapes([s1, s2])
    assert batched.is_batch is True
    assert batched.n_batch == 2
    assert batched.array.ndim == 2


def test_collate_shapes_empty_raises(backend):
    with pytest.raises(ValueError):
        collate_shapes([])


def test_separate_shapes_non_batched_raises(backend):
    e = get_fixed_edge(backend=backend)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"edge": e}, non_fictitious=np.ones(2), backend=backend)
    with pytest.raises(ValueError):
        separate_shapes(gs)


def test_collate_and_separate_roundtrip(backend):
    e = get_fixed_edge(backend=backend)
    gs1 = GraphShape.from_dict(hyper_edge_set_dict={"t": e, "u": e}, non_fictitious=np.ones(2), backend=backend)
    gs2 = GraphShape.from_dict(hyper_edge_set_dict={"t": e, "u": e}, non_fictitious=np.ones(4), backend=backend)
    batched = collate_shapes([gs1, gs2])
    separated = separate_shapes(batched)
    assert isinstance(separated, list)
    assert len(separated) == 2
    assert set(separated[0].hyper_edge_sets.keys()) == set(batched.hyper_edge_sets.keys())
    assert separated[0].addresses.shape == ()
    assert separated[1].addresses.shape == ()


def test_collate_preserves_type(backend):
    e = get_fixed_edge(backend=backend)
    gs1 = GraphShape.from_dict(hyper_edge_set_dict={"e": e}, non_fictitious=np.ones(2), backend=backend)
    gs2 = GraphShape.from_dict(hyper_edge_set_dict={"e": e}, non_fictitious=np.ones(4), backend=backend)
    batched = collate_shapes([gs1, gs2])
    assert type(batched) is type(gs1)


def test_max_and_sum_binary_operations(backend):
    e = get_fixed_edge(backend=backend)
    gs1 = GraphShape.from_dict(hyper_edge_set_dict={"A": e, "B": e}, non_fictitious=np.ones(3), backend=backend)
    xp = backend.xp
    gs2 = GraphShape(backend=backend, hyper_edge_sets={"A": xp.array(5), "B": xp.array(1)}, addresses=xp.array(10))

    m = GraphShape.max(gs1, gs2)
    assert int(m.hyper_edge_sets["A"]) == max(int(gs1.hyper_edge_sets["A"]), int(gs2.hyper_edge_sets["A"]))
    assert int(m.addresses) == max(int(gs1.addresses), int(gs2.addresses))

    s = GraphShape.sum(gs1, gs2)
    assert int(s.hyper_edge_sets["A"]) == int(gs1.hyper_edge_sets["A"]) + int(gs2.hyper_edge_sets["A"])
    assert int(s.addresses) == int(gs1.addresses) + int(gs2.addresses)


def test_max_shape_and_sum_shapes_list_ops_and_errors(backend):
    e = get_fixed_edge(backend=backend)
    gs1 = GraphShape.from_dict(hyper_edge_set_dict={"X": e}, non_fictitious=np.ones(2), backend=backend)
    xp = backend.xp
    gs2 = GraphShape(backend=backend, hyper_edge_sets={"X": xp.array(5)}, addresses=xp.array(10))

    max_res = max_shape([gs1, gs2])
    assert isinstance(max_res, GraphShape)
    assert int(max_res.hyper_edge_sets["X"]) == max(int(gs1.hyper_edge_sets["X"]), int(gs2.hyper_edge_sets["X"]))

    sum_res = sum_shapes([gs1, gs2])
    assert int(sum_res.hyper_edge_sets["X"]) == int(gs1.hyper_edge_sets["X"]) + int(gs2.hyper_edge_sets["X"])

    with pytest.raises(ValueError):
        max_shape([])
    with pytest.raises(ValueError):
        sum_shapes([])
    with pytest.raises(ValueError):
        max_shape([gs1, "not a graphshape"])


def test_to_backend_conversion(backend):
    from energnn.graph.backend import JaxBackend, NumpyBackend

    e = get_fixed_edge(backend=backend)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"e": e}, non_fictitious=np.ones(3), backend=backend)
    other = NumpyBackend() if isinstance(backend, JaxBackend) else JaxBackend()
    converted = gs.to_backend(other)
    assert converted._backend == other
    assert int(converted.addresses) == int(gs.addresses)
