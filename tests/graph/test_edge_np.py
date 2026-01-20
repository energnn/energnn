#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import pandas as pd
import pytest

from energnn.graph.edge import (
    Edge,
    collate_edges,
    separate_edges,
    concatenate_edges,
    check_dict_shape,
    build_edge_shape,
    dict2array,
    check_dict_or_none,
    check_no_nan,
    check_valid_addresses,
)
from tests.graph.utils import get_fixed_edge


def test_from_dict_and_basic_props():
    edge = get_fixed_edge()
    # array concatenation shape: (n_obj, n_feats + n_ports) -> (2, 2 + 2) = (2,4)
    assert edge.array.shape == (2, 4)
    assert edge.is_single is True
    assert edge.is_batch is False
    assert edge.n_obj == 2
    # feature_dict returns same arrays
    fd = edge.feature_dict
    assert "b" in fd and "w" in fd
    np.testing.assert_allclose(fd["b"], np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(fd["w"], np.array([0.5, 1.0], dtype=np.float32))


def test_feature_flat_array_getter_and_setter():
    edge = get_fixed_edge()
    flat = edge.feature_flat_array
    # For single: 1D length n_obj * n_feats = 2 * 2 = 4
    assert flat.shape == (4,)
    # Create new flat array with same shape, set it, and verify feature_array changed accordingly
    new_flat = flat + 1.0
    edge.feature_flat_array = new_flat
    # reshape Fortran order: first column then second column -> check sum changed
    np.testing.assert_allclose(edge.feature_array, new_flat.reshape([edge.n_obj, -1], order="F"))
    # the setter should not raise when shapes match; check final flat matches
    np.testing.assert_allclose(edge.feature_flat_array, new_flat)


def test_feature_flat_array_setter_shape_mismatch_raises():
    edge = get_fixed_edge()
    flat = edge.feature_flat_array
    with pytest.raises(ValueError):
        # wrong shape
        edge.feature_flat_array = flat[:-1]


def test_pad_and_unpad_single():
    edge = get_fixed_edge()
    old_n = edge.n_obj
    edge.pad(4)
    assert edge.n_obj == 4
    # feature padding should add rows (zeros)
    assert edge.feature_array.shape[0] == 4
    # addresses padded too
    for k, v in edge.address_dict.items():
        assert v.shape[0] == 4
    # unpad back to original
    edge.unpad(old_n)
    assert edge.n_obj == old_n
    assert edge.feature_array.shape[0] == old_n


def test_pad_on_batch_raises():
    e1 = get_fixed_edge()
    e2 = get_fixed_edge()
    batch = collate_edges([e1, e2])
    with pytest.raises(ValueError):
        batch.pad(10)


def test_unpad_on_batch_raises():
    e1 = get_fixed_edge()
    e2 = get_fixed_edge()
    batch = collate_edges([e1, e2])
    with pytest.raises(ValueError):
        batch.unpad(1)


def test_offset_addresses():
    edge = get_fixed_edge()
    orig = {k: v.copy() for k, v in edge.address_dict.items()}
    edge.offset_addresses(10)
    for k in orig:
        np.testing.assert_allclose(edge.address_dict[k], orig[k] + 10)


def test_collate_and_separate_roundtrip():
    e1 = get_fixed_edge()
    # modify e2 slightly so we can check separation
    e2 = get_fixed_edge()
    e2.offset_addresses(100)
    batch = collate_edges([e1, e2])
    assert batch.is_batch is True
    assert batch.n_batch == 2
    separated = separate_edges(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # first separated edge should equal e1 (addresses, features)
    np.testing.assert_array_equal(separated[0].address_dict["dst"], e1.address_dict["dst"])
    np.testing.assert_array_equal(separated[1].address_dict["dst"], e2.address_dict["dst"])


def test_collate_empty_raises():
    with pytest.raises(IndexError):
        collate_edges([])


def test_collate_inconsistent_keys_raises():
    e1 = get_fixed_edge()
    e2 = get_fixed_edge()
    # remove addresses from e2 to produce mismatch
    e2.address_dict = None
    with pytest.raises(ValueError):
        collate_edges([e1, e2])


def test_concatenate_edges():
    e1 = get_fixed_edge()
    e2 = get_fixed_edge()
    concatenated = concatenate_edges([e1, e2])
    # n_obj should be sum
    assert concatenated.n_obj == e1.n_obj + e2.n_obj
    # addresses concatenated
    np.testing.assert_array_equal(concatenated.address_dict["dst"][:2], e1.address_dict["dst"])
    np.testing.assert_array_equal(concatenated.address_dict["dst"][2:], e2.address_dict["dst"])


def test_check_dict_shape_and_build_edge_shape_errors():
    # mismatching last dims
    d1 = {"a": np.zeros((3,), dtype=np.float32), "b": np.zeros((4,), dtype=np.float32)}
    with pytest.raises(ValueError):
        check_dict_shape(d=d1, n_objects=None)
    # build_edge_shape with both None should raise
    with pytest.raises(ValueError):
        build_edge_shape(address_dict=None, feature_dict=None)


def test_dict2array_and_sorting():
    d = {"z": np.array([1, 2], dtype=np.float32), "a": np.array([3, 4], dtype=np.float32)}
    arr = dict2array(d)
    # keys sorted -> ['a', 'z'] -> columns correspond
    np.testing.assert_array_equal(arr[:, 0], d["a"])
    np.testing.assert_array_equal(arr[:, 1], d["z"])
    assert dict2array(None) is None


def test_check_dict_or_none_and_nan_and_valid_addresses():
    # non-dict non-None raises
    with pytest.raises(ValueError):
        check_dict_or_none(np.array([1, 2, 3]))
    # NaN detection in address
    with pytest.raises(ValueError):
        check_no_nan(address_dict={"a": np.array([0.0, np.nan], dtype=np.float32)}, feature_dict=None)
    # NaN detection in feature
    with pytest.raises(ValueError):
        check_no_nan(address_dict=None, feature_dict={"f": np.array([np.nan, 1.0], dtype=np.float32)})
    # valid addresses: float values but integer-valued pass (1.0)
    check_valid_addresses({"x": np.array([1.0, 2.0], dtype=np.float32)})
    # non-integer valued should raise
    with pytest.raises(ValueError):
        check_valid_addresses({"x": np.array([1.0, 2.3], dtype=np.float32)})


def test_str_repr_single_and_batch():
    e = get_fixed_edge()
    s = str(e)
    assert "features" in s and "addresses" in s
    # batch
    b = collate_edges([e, e])
    s2 = str(b)
    assert "features" in s2 and "addresses" in s2
