#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import copy

import jax
import numpy as np

from energnn.graph import Edge, collate_edges, separate_edges
from energnn.graph.jax import JaxEdge
from tests.utils import sample_edge


def test_edge_from_dict():

    address_dict = {"a0": np.array([0, 1]), "a1": np.array([2, 3])}
    feature_dict = {"f0": np.array([4.0, 5.0]), "f1": np.array([6.0, 7.0])}

    edge = Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 2

    edge = Edge.from_dict(address_dict=address_dict, feature_dict=None)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 2

    edge = Edge.from_dict(address_dict=None, feature_dict=feature_dict)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 2

    try:
        Edge.from_dict(address_dict=None, feature_dict=None)
        raise ValueError
    except:
        pass

    try:
        address_dict = {"a0": np.array([0]), "a1": np.array([2, 3])}
        feature_dict = {"f0": np.array([4.0, 5.0]), "f1": np.array([6.0, 7.0])}
        Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)
        raise ValueError
    except:
        pass

    try:
        address_dict = {"a0": np.array([0, 1]), "a1": np.array([2, 3])}
        feature_dict = {"f0": np.array([4.0]), "f1": np.array([6.0, 7.0])}
        Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)
        raise ValueError
    except:
        pass

    try:
        address_dict = {"a0": np.array([0, 1]), "a1": np.array([2, 3])}
        feature_dict = {"f0": np.array([4.0]), "f1": np.array([6.0])}
        Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)
        raise ValueError
    except:
        pass

    # Tester aussi s'il y a une inconsistance dans le nombre d'objets


def test_edge_creation():

    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 3
    np.testing.assert_equal(edge.non_fictitious, np.ones(3))

    edge = sample_edge(n_obj=3, feature_list=[], address_list=["a0", "a1"])
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 3
    np.testing.assert_equal(edge.non_fictitious, np.ones(3))

    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=[])
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 3
    np.testing.assert_equal(edge.non_fictitious, np.ones(3))

    edge = sample_edge(n_obj=0, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 0
    np.testing.assert_equal(edge.non_fictitious, np.ones(0))


def test_edge_padding():

    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    edge_copy = copy.deepcopy(edge)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 3

    # Pad with wrong value
    try:
        edge.pad(target_shape=2)
        raise AssertionError
    except ValueError:
        pass

    # Pad with same value
    edge.pad(target_shape=3)

    # Pad with larger value
    edge.pad(target_shape=5)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 5

    # Unpad with larger value
    try:
        edge.unpad(target_shape=7)
        raise AssertionError
    except ValueError:
        pass

    # Unpad with same value
    edge.unpad(target_shape=5)

    # Test with unpadding
    edge.unpad(target_shape=3)
    assert edge.is_single
    assert not edge.is_batch
    assert edge.n_obj == 3

    np.testing.assert_equal(edge, edge_copy)


def test_edge_collating():

    edge_0 = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    edge_0_copy = copy.deepcopy(edge_0)
    edge_1 = sample_edge(n_obj=0, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    edge_1_copy = copy.deepcopy(edge_1)

    try:
        collate_edges([edge_0, edge_1])
        raise AssertionError
    except ValueError:
        pass

    edge_0.pad(target_shape=5)
    edge_1.pad(target_shape=5)

    edge_batch = collate_edges([edge_0, edge_1])
    assert edge_batch.is_batch
    assert not edge_batch.is_single
    assert edge_batch.n_obj == 5
    assert edge_batch.n_batch == 2

    jax_edge_batch = JaxEdge.from_numpy_edge(edge_batch)
    new_edge_batch = jax_edge_batch.to_numpy_edge()

    edge_list = separate_edges(new_edge_batch)
    edge_0, edge_1 = edge_list[0], edge_list[1]
    edge_0.unpad(target_shape=3)
    np.testing.assert_equal(edge_0, edge_0_copy)
    edge_1.unpad(target_shape=0)
    np.testing.assert_equal(edge_1, edge_1_copy)


def test_edge_collating_failure():
    edge_0 = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    edge_1 = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0"])
    edge_2 = sample_edge(n_obj=3, feature_list=["f0"], address_list=["a0", "a1"])

    try:
        collate_edges([edge_0, edge_1])
        raise AssertionError
    except:
        pass

    try:
        collate_edges([edge_0, edge_2])
        raise AssertionError
    except:
        pass

    try:
        collate_edges([edge_2, edge_1])
        raise AssertionError
    except:
        pass


def test_jax_64():
    jax.config.update("jax_enable_x64", True)
    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    jax_edge = JaxEdge.from_numpy_edge(edge, device=jax.devices("cpu")[0], dtype="float64")
    assert jax_edge.feature_flat_array.dtype == jax.numpy.float64
    new_edge = jax_edge.to_numpy_edge()
    np.testing.assert_equal(edge, new_edge)


def test_jax_32():
    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    jax_edge = JaxEdge.from_numpy_edge(edge, device=jax.devices("cpu")[0], dtype="float32")
    assert jax_edge.feature_flat_array.dtype == jax.numpy.float32
    new_edge = jax_edge.to_numpy_edge()
    np.testing.assert_equal(edge, new_edge)


def test_jax_16():
    edge = sample_edge(n_obj=3, feature_list=["f0", "f1"], address_list=["a0", "a1"])
    jax_edge = JaxEdge.from_numpy_edge(edge, device=jax.devices("cpu")[0], dtype="float16")
    assert jax_edge.feature_flat_array.dtype == jax.numpy.float16
