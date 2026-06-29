#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
import numpy as np

import energnn.model.normalizer.tdigest_normalizer as tdn
from energnn.graph import GraphStructure, HyperEdgeSetStructure, Graph, HyperEdgeSet, JaxBackend
from energnn.model.normalizer.tdigest_normalizer import (
    TDigestModule,
    TDigestNormalizer,
)
from energnn.problem.example import LinearSystemProblemLoader

# make deterministic
np.random.seed(0)

# small fixture graphs (used by some tests)
n = 10
pb_loader = LinearSystemProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)  # single example usable in tests


def test_tdigest_module_init():
    in_size = 4
    module = TDigestModule(in_size=in_size, update_limit=10, n_breakpoints=20, max_centroids=100, use_running_average=False)
    assert module.in_size == in_size
    assert module.updates[...] == 0
    assert module.xp_var.shape == (20, in_size)
    assert module.slopes_var.shape == (2, in_size)
    # JaxTDigest internal capacity is 1.5 * max_centroids
    assert module.digest.get_value().centroids.shape == (in_size, 150, 2)


def test_tdigest_module_update():
    in_size = 2
    module = TDigestModule(in_size=in_size, update_limit=3, n_breakpoints=10, max_centroids=10, use_running_average=False)

    # First update
    x = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mask = jnp.ones((3, 1))
    _ = module(x, mask)

    assert module.updates[...] == 1
    # Check that xp_var has been updated from its initial linear linspace
    assert not jnp.allclose(module.xp_var[...], jnp.tile(jnp.linspace(-1.0, 1.0, 10)[:, None], (1, in_size)))

    # Multiple updates until limit
    for _ in range(5):
        module(x, mask)

    assert module.updates[...] == 3  # Should stop at update_limit


def test_tdigest_module_normalization_range():
    jax.config.update("jax_enable_x64", True)
    try:
        in_size = 1
        module = TDigestModule(
            in_size=in_size, update_limit=100, n_breakpoints=50, max_centroids=100, use_running_average=False
        )

        # Train on a normal distribution
        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (1000, 1))
        mask = jnp.ones((1000, 1))

        # Several updates to converge
        for _ in range(5):
            module(data, mask)

        # Test normalization
        test_data = jnp.array([[-3.0], [0.0], [3.0]])
        norm_out = module(test_data, jnp.ones((3, 1)))

        # Values should be roughly in [-1, 1] for data within the seen distribution
        # Median (0.0) should be around 0.0
        assert jnp.abs(norm_out[1, 0]) < 0.1
        # Extremes should be close to -1 and 1
        assert norm_out[0, 0] < -0.95
        assert norm_out[2, 0] > 0.95
    finally:
        jax.config.update("jax_enable_x64", False)


def test_tdigest_module_inference_mode():
    module = TDigestModule(in_size=1, update_limit=10, n_breakpoints=10, max_centroids=10, use_running_average=True)
    initial_xp = module.xp_var[...]

    data = jnp.array([[1.0], [2.0], [3.0]])
    module(data, jnp.ones((3, 1)))

    assert module.updates[...] == 0
    assert jnp.all(module.xp_var[...] == initial_xp)


def test_tdigest_module_masking():
    module = TDigestModule(in_size=1, update_limit=10, n_breakpoints=10, max_centroids=10, use_running_average=False)

    # Only the first element is non-fictitious
    data = jnp.array([[100.0], [0.0]])
    mask = jnp.array([[1.0], [0.0]])

    module(data, mask)

    # Min/Max should reflect only the first element
    assert jnp.allclose(module.min_var, 100.0)
    assert jnp.allclose(module.max_var, 100.0)


def test_tdigest_module_batch_shape():
    in_size = 2
    module = TDigestModule(in_size=in_size, update_limit=10, n_breakpoints=10, max_centroids=10, use_running_average=False)

    # Batch of 2 examples, each with 3 nodes
    data = jnp.ones((2, 3, 2))
    mask = jnp.ones((2, 3, 1))

    out = module(data, mask)
    assert out.shape == (2, 3, 2)
    assert module.updates[...] == 1


def test_tdigest_normalizer_init():
    struct = GraphStructure(
        hyper_edge_sets={
            "nodes": HyperEdgeSetStructure(port_list=["id"], feature_list=["a", "b"]),
            "edges": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None),
        }
    )
    normalizer = TDigestNormalizer(struct, update_limit=10)
    assert "nodes" in normalizer.module_dict
    assert isinstance(normalizer.module_dict["nodes"], TDigestModule)
    assert normalizer.module_dict["edges"] is None


def test_tdigest_normalizer_call():
    # Use jax_context and context_structure from the file setup
    struct = pb_batch.context_structure
    normalizer = TDigestNormalizer(struct, update_limit=10)

    # Call normalizer
    norm_graph, info = normalizer(graph=jax_context, get_info=True)

    assert isinstance(norm_graph, Graph)
    assert "input_graph" in info
    assert "output_graph" in info

    # Check that some values changed in nodes features
    for k in jax_context.hyper_edge_sets:
        if normalizer.module_dict[k] is not None:
            original = jax_context.hyper_edge_sets[k].feature_array
            normalized = norm_graph.hyper_edge_sets[k].feature_array
            if original is not None and original.shape[-2] > 0:
                assert not jnp.allclose(original, normalized)


def test_tdigest_normalizer_set_running_average():
    struct = pb_batch.context_structure
    normalizer = TDigestNormalizer(struct, update_limit=10)

    normalizer.set_running_average(True)
    assert normalizer.use_running_average is True
    for module in normalizer.module_dict.values():
        if module is not None:
            assert module.use_running_average is True


def test_tdigest_normalizer_multi_sets():
    struct = GraphStructure(
        hyper_edge_sets={
            "s1": HyperEdgeSetStructure(port_list=["p1"], feature_list=["f1"]),
            "s2": HyperEdgeSetStructure(port_list=["p2"], feature_list=["f2", "f3"]),
            "s3": HyperEdgeSetStructure(port_list=["p3"], feature_list=None),
        }
    )
    normalizer = TDigestNormalizer(struct, update_limit=10)

    # Create a dummy graph
    h1 = HyperEdgeSet(
        backend=JaxBackend(),
        feature_array=jnp.array([[1.0], [2.0]]),
        feature_names=["f1"],
        non_fictitious=jnp.array([True, True]),
        port_dict={"p1": jnp.array([0, 1])},
    )
    h2 = HyperEdgeSet(
        backend=JaxBackend(),
        feature_array=jnp.array([[10.0, 100.0]]),
        feature_names=["f2", "f3"],
        non_fictitious=jnp.array([True]),
        port_dict={"p2": jnp.array([0])},
    )
    h3 = HyperEdgeSet(
        backend=JaxBackend(),
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.array([True]),
        port_dict={"p3": jnp.array([0])},
    )

    graph = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"s1": h1, "s2": h2, "s3": h3},
        non_fictitious_addresses=jnp.array([0, 1]),
        true_shape=None,  # Not used here
        current_shape=None,
    )

    norm_graph, _ = normalizer(graph=graph)

    # Check that s1 and s2 are normalized
    assert not jnp.allclose(norm_graph.hyper_edge_sets["s1"].feature_array, h1.feature_array)
    assert not jnp.allclose(norm_graph.hyper_edge_sets["s2"].feature_array, h2.feature_array)
    # Check that s3 is unchanged
    assert norm_graph.hyper_edge_sets["s3"].feature_array is None


def test_tdigest_normalizer_update_limit():
    struct = GraphStructure(hyper_edge_sets={"s1": HyperEdgeSetStructure(port_list=["p1"], feature_list=["f1"])})
    normalizer = TDigestNormalizer(struct, update_limit=2)

    h1 = HyperEdgeSet(
        feature_array=jnp.array([[1.0]]),
        feature_names=["f1"],
        non_fictitious=jnp.array([True]),
        port_dict={"p1": jnp.array([0])},
    )
    graph = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"s1": h1},
        non_fictitious_addresses=jnp.array([0]),
        true_shape=None,
        current_shape=None,
    )

    # Call 3 times
    normalizer(graph=graph)
    normalizer(graph=graph)
    normalizer(graph=graph)

    assert normalizer.module_dict["s1"].updates[...] == 2
