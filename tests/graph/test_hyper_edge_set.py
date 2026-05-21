# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.hyper_edge_set import (
    HyperEdgeSet,
    build_hyper_edge_set_shape,
    check_dict_or_none,
    check_dict_shape,
    check_no_nan,
    check_valid_ports,
    collate_hyper_edge_sets,
    concatenate_hyper_edge_sets,
    dict2array,
    separate_hyper_edge_sets,
)
from tests.graph.utils import get_fixed_edge


# ---------------------------------------------------------------------------
# Backend-parametrized tests (run with both NumpyBackend and JaxBackend)
# ---------------------------------------------------------------------------


def test_from_dict_and_basic_props(backend):
    edge = get_fixed_edge(backend=backend)
    assert edge.array.shape == (2, 4)
    assert edge.is_single is True
    assert edge.is_batch is False
    assert edge.n_obj == 2
    fd = edge.feature_dict
    assert "b" in fd and "w" in fd
    np.testing.assert_allclose(np.array(fd["b"]), np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(np.array(fd["w"]), np.array([0.5, 1.0], dtype=np.float32))


def test_backend_stored_correctly(backend):
    edge = get_fixed_edge(backend=backend)
    assert edge._backend == backend


def test_feature_flat_array_getter_and_setter(backend):
    edge = get_fixed_edge(backend=backend)
    flat = edge.feature_flat_array
    assert flat.shape == (4,)
    xp = backend.xp
    new_flat = xp.array(np.array(flat) + 1.0)
    edge.feature_flat_array = new_flat
    np.testing.assert_allclose(np.array(edge.feature_flat_array), np.array(new_flat))


def test_feature_flat_array_setter_shape_mismatch_raises(backend):
    edge = get_fixed_edge(backend=backend)
    flat = edge.feature_flat_array
    with pytest.raises(ValueError):
        xp = backend.xp
        edge.feature_flat_array = xp.array(np.array(flat)[:-1])


def test_pad_and_unpad_single(backend):
    edge = get_fixed_edge(backend=backend)
    old_n = edge.n_obj
    edge.pad(4)
    assert edge.n_obj == 4
    assert edge.feature_array.shape[0] == 4
    for v in edge.port_dict.values():
        assert v.shape[0] == 4
    edge.unpad(old_n)
    assert edge.n_obj == old_n
    assert edge.feature_array.shape[0] == old_n


def test_pad_on_batch_raises(backend):
    batch = collate_hyper_edge_sets([get_fixed_edge(backend=backend), get_fixed_edge(backend=backend)])
    with pytest.raises(ValueError):
        batch.pad(10)


def test_unpad_on_batch_raises(backend):
    batch = collate_hyper_edge_sets([get_fixed_edge(backend=backend), get_fixed_edge(backend=backend)])
    with pytest.raises(ValueError):
        batch.unpad(1)


def test_offset_addresses(backend):
    edge = get_fixed_edge(backend=backend)
    orig = {k: np.array(v).copy() for k, v in edge.port_dict.items()}
    edge.offset_addresses(10)
    for k in orig:
        np.testing.assert_allclose(np.array(edge.port_dict[k]), orig[k] + 10)


def test_collate_and_separate_roundtrip(backend):
    e1 = get_fixed_edge(backend=backend)
    e2 = get_fixed_edge(backend=backend)
    e2.offset_addresses(100)
    batch = collate_hyper_edge_sets([e1, e2])
    assert batch.is_batch is True
    assert batch.n_batch == 2
    separated = separate_hyper_edge_sets(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    np.testing.assert_array_equal(np.array(separated[0].port_dict["dst"]), np.array(e1.port_dict["dst"]))
    np.testing.assert_array_equal(np.array(separated[1].port_dict["dst"]), np.array(e2.port_dict["dst"]))


def test_collate_preserves_subclass_type(backend):
    e1 = get_fixed_edge(backend=backend)
    e2 = get_fixed_edge(backend=backend)
    batch = collate_hyper_edge_sets([e1, e2])
    assert type(batch) is type(e1)


def test_collate_empty_raises(backend):
    with pytest.raises(IndexError):
        collate_hyper_edge_sets([])


def test_collate_inconsistent_keys_raises(backend):
    e1 = get_fixed_edge(backend=backend)
    e2 = get_fixed_edge(backend=backend)
    e2.port_dict = None
    with pytest.raises(ValueError):
        collate_hyper_edge_sets([e1, e2])


def test_concatenate_edges(backend):
    e1 = get_fixed_edge(backend=backend)
    e2 = get_fixed_edge(backend=backend)
    cat = concatenate_hyper_edge_sets([e1, e2])
    assert cat.n_obj == e1.n_obj + e2.n_obj
    np.testing.assert_array_equal(np.array(cat.port_dict["dst"])[:2], np.array(e1.port_dict["dst"]))
    np.testing.assert_array_equal(np.array(cat.port_dict["dst"])[2:], np.array(e2.port_dict["dst"]))


def test_str_repr_single_and_batch(backend):
    e = get_fixed_edge(backend=backend)
    s = str(e)
    assert "features" in s and "ports" in s
    b = collate_hyper_edge_sets([e, e])
    s2 = str(b)
    assert "features" in s2 and "ports" in s2


def test_to_backend_conversion(backend):
    from energnn.graph.backend import JaxBackend, NumpyBackend

    e = get_fixed_edge(backend=backend)
    # Round-trip through the other backend
    other = NumpyBackend() if isinstance(backend, JaxBackend) else JaxBackend()
    converted = e.to_backend(other)
    assert converted._backend == other
    np.testing.assert_allclose(np.array(converted.feature_array), np.array(e.feature_array))


# ---------------------------------------------------------------------------
# Validation helpers (backend-agnostic, always numpy)
# ---------------------------------------------------------------------------


def test_check_dict_shape_mismatch_raises():
    d = {"a": np.zeros(3, dtype=np.float32), "b": np.zeros(4, dtype=np.float32)}
    with pytest.raises(ValueError):
        check_dict_shape(d=d, n_objects=None)


def test_build_edge_shape_both_none_raises():
    with pytest.raises(ValueError):
        build_hyper_edge_set_shape(port_dict=None, feature_dict=None)


def test_dict2array_sorting():
    d = {"z": np.array([1, 2], dtype=np.float32), "a": np.array([3, 4], dtype=np.float32)}
    arr = dict2array(d)
    np.testing.assert_array_equal(arr[:, 0], d["a"])
    np.testing.assert_array_equal(arr[:, 1], d["z"])
    assert dict2array(None) is None


def test_check_dict_or_none_non_dict_raises():
    with pytest.raises(ValueError):
        check_dict_or_none(np.array([1, 2, 3]))


def test_check_no_nan_port_raises():
    with pytest.raises(ValueError):
        check_no_nan(port_dict={"a": np.array([0.0, np.nan])}, feature_dict=None)


def test_check_no_nan_feature_raises():
    with pytest.raises(ValueError):
        check_no_nan(port_dict=None, feature_dict={"f": np.array([np.nan, 1.0])})


def test_check_valid_ports_integer_pass():
    check_valid_ports({"x": np.array([1.0, 2.0], dtype=np.float32)})


def test_check_valid_ports_non_integer_raises():
    with pytest.raises(ValueError):
        check_valid_ports({"x": np.array([1.0, 2.3], dtype=np.float32)})
