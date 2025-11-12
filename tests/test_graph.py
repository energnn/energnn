#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import copy

import jax
import numpy as np

from energnn.graph import collate_graphs, concatenate_graphs, max_shape, separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import build_shape, sample_graph


def test_edge_creation():

    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )

    try:
        graph = sample_graph(
            edge_params={
                "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
                "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
            },
            n_addr=4,
        )
    except AssertionError:
        pass


def test_padding():

    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    graph_copy = copy.deepcopy(graph)

    target_shape = build_shape(edge_params={"node": 6, "edge": 7}, n_addr=6)
    graph.pad(target_shape=target_shape)
    np.testing.assert_equal(graph.current_shape, target_shape)
    np.testing.assert_equal(graph.true_shape, graph_copy.current_shape)
    graph.unpad()
    np.testing.assert_equal(graph, graph_copy)
    assert graph.is_single
    assert not graph.is_batch


def test_padding_zero_object():

    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 0, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 0, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=0,
    )
    graph_copy = copy.deepcopy(graph)

    target_shape = build_shape(edge_params={"node": 6, "edge": 7}, n_addr=6)
    graph.pad(target_shape=target_shape)
    np.testing.assert_equal(graph.current_shape, target_shape)
    np.testing.assert_equal(graph.true_shape, graph_copy.current_shape)
    graph.unpad()
    np.testing.assert_equal(graph, graph_copy)
    assert graph.is_single
    assert not graph.is_batch


def test_padding_failure():
    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )

    target_shape = build_shape(edge_params={"node": 3, "edge": 5}, n_addr=5)
    graph.pad(target_shape=target_shape)
    np.testing.assert_equal(graph.current_shape, graph.true_shape)

    try:
        target_shape = build_shape(edge_params={"node": 3, "edge": 5}, n_addr=4)
        graph.pad(target_shape=target_shape)
        raise AssertionError
    except:
        pass

    try:
        target_shape = build_shape(edge_params={"node": 2, "edge": 5}, n_addr=5)
        graph.pad(target_shape=target_shape)
        raise AssertionError
    except:
        pass

    try:
        target_shape = build_shape(edge_params={"node": 3, "edge": 4}, n_addr=5)
        graph.pad(target_shape=target_shape)
        raise AssertionError
    except:
        pass

    try:
        target_shape = build_shape(edge_params={"node": 3}, n_addr=5)
        graph.pad(target_shape=target_shape)
        raise AssertionError
    except:
        pass


def test_collating():
    graph_0 = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b", "e"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    graph_0_copy = copy.deepcopy(graph_0)
    graph_1 = sample_graph(
        edge_params={
            "node": {"n_obj": 2, "feature_list": ["a", "b", "e"], "address_list": ["0"]},
            "edge": {"n_obj": 6, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=7,
    )
    graph_1_copy = copy.deepcopy(graph_1)
    target_shape = max_shape([graph_0.current_shape, graph_1.current_shape])

    graph_0.pad(target_shape=target_shape)
    graph_1.pad(target_shape=target_shape)
    graph_batch = collate_graphs([graph_0, graph_1])
    assert graph_batch.is_batch
    assert not graph_batch.is_single

    jax_graph_batch = JaxGraph.from_numpy_graph(graph_batch)
    new_edge_batch = jax_graph_batch.to_numpy_graph()

    graph_list = separate_graphs(new_edge_batch)
    graph_0, graph_1 = graph_list[0], graph_list[1]
    graph_0.unpad()
    np.testing.assert_equal(graph_0_copy, graph_0)
    graph_1.unpad()
    np.testing.assert_equal(graph_1_copy, graph_1)


def test_collating_failure():
    graph_0 = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    graph_1 = sample_graph(
        edge_params={
            "node": {"n_obj": 0, "feature_list": ["a", "b"], "address_list": ["0"]},
        },
        n_addr=5,
    )
    graph_2 = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    graph_3 = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["1"]},
        },
        n_addr=5,
    )

    try:
        collate_graphs([graph_0, graph_1])
        raise AssertionError
    except:
        pass

    try:
        collate_graphs([graph_0, graph_2])
        raise AssertionError
    except:
        pass

    try:
        collate_graphs([graph_0, graph_3])
        raise AssertionError
    except:
        pass


def test_concatenating():
    graph_0 = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b", "e"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    graph_1 = sample_graph(
        edge_params={
            "node": {"n_obj": 2, "feature_list": ["a", "b", "e"], "address_list": ["0"]},
            "edge": {"n_obj": 6, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=6,
    )
    graph = concatenate_graphs([graph_0, graph_1])
    assert graph.is_single
    assert not graph.is_batch
    # TODO rajouter des choses


def test_jax_16():
    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    jax_graph = JaxGraph.from_numpy_graph(graph, device=jax.devices("cpu")[0], dtype="float16")
    assert jax_graph.feature_flat_array.dtype == jax.numpy.float16


def test_jax_32():
    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    jax_graph = JaxGraph.from_numpy_graph(graph, device=jax.devices("cpu")[0], dtype="float32")
    assert jax_graph.feature_flat_array.dtype == jax.numpy.float32


def test_jax_64():
    jax.config.update("jax_enable_x64", True)
    graph = sample_graph(
        edge_params={
            "node": {"n_obj": 3, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 5, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        n_addr=5,
    )
    jax_graph = JaxGraph.from_numpy_graph(graph, device=jax.devices("cpu")[0], dtype="float64")
    assert jax_graph.feature_flat_array.dtype == jax.numpy.float64


# Idée checker que tout marche quand il n'y a pas d'objet d'une classe.

# vérifier que le padding ne marche pas si on a une incompatibilité de dim / feature / adress / object name / etc.
