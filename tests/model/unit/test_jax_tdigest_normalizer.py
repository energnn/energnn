#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax.numpy as jnp
import numpy as np
from energnn.model.normalizer.jax_tdigest_normalizer import JAXTDigestModule, JAXTDigestNormalizer
from energnn.graph import GraphStructure


def test_jax_tdigest_module_basic():
    in_size = 4
    update_limit = 10
    n_breakpoints = 11
    max_centroids = 100

    module = JAXTDigestModule(in_size, update_limit, n_breakpoints, max_centroids, use_running_average=False)

    # First batch
    x = jnp.array(np.random.normal(0, 1, (128, in_size)), dtype=jnp.float32)
    mask = jnp.ones((128, 1), dtype=bool)

    out = module(x, mask)
    assert out.shape == x.shape
    assert module.updates[...] == 1

    # Check that xp and fp are updated
    assert not jnp.all(module.xp_var[...] == jnp.linspace(-1, 1, n_breakpoints)[:, None])


def test_jax_tdigest_module_accuracy():
    # Similar to the benchmark but as a test
    in_size = 1
    n_breakpoints = 101
    max_centroids = 1000
    module = JAXTDigestModule(in_size, 100, n_breakpoints, max_centroids, use_running_average=False)

    # Inject 10000 points from Normal(10, 2)
    np.random.seed(42)
    data = np.random.normal(10, 2, (1000, 1)).astype(np.float32)
    mask = np.ones((1000, 1), dtype=bool)

    module(jnp.array(data), jnp.array(mask))

    # Check median (index 50 for 101 points)
    median = module.xp_var[50, 0]
    assert np.abs(median - 10.0) < 0.5

    # Check min/max
    assert np.abs(module.min_var[0] - data.min()) < 0.1
    assert np.abs(module.max_var[0] - data.max()) < 0.1


def test_jax_tdigest_normalizer_integration():
    from energnn.graph.jax import JaxGraph, JaxHyperEdgeSet
    from energnn.graph import HyperEdgeSetStructure

    struct = GraphStructure(hyper_edge_sets={"edges": HyperEdgeSetStructure(feature_list=["a", "b"], port_list=[])})

    norm = JAXTDigestNormalizer(struct, update_limit=10)

    feat = jnp.array(np.random.normal(0, 1, (1, 10, 2)), dtype=jnp.float32)
    non_fict = jnp.ones((1, 10), dtype=bool)

    graph = JaxGraph(
        hyper_edge_sets={
            "edges": JaxHyperEdgeSet(feature_array=feat, feature_names=["a", "b"], non_fictitious=non_fict, port_dict={})
        },
        non_fictitious_addresses={},
        true_shape={},
        current_shape={},
    )

    norm_graph, info = norm(graph=graph)
    assert norm_graph.hyper_edge_sets["edges"].feature_array is not None
    assert norm_graph.hyper_edge_sets["edges"].feature_array.shape == feat.shape
