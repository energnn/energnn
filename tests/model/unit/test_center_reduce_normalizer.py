#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import logging
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from energnn.graph import GraphStructure, Graph, HyperEdgeSet, HyperEdgeSetStructure, JaxBackend
from energnn.model.normalizer.center_reduce_normalizer import (
    HyperEdgeSetCenterReduceNormalizer,
    CenterReduceNormalizer,
)
from energnn.problem.example import LinearSystemProblemLoader

# make deterministic
np.random.seed(0)

# small fixture graphs (used by some tests)
n = 10
pb_loader = LinearSystemProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)  # single example usable


def test_hyper_edge_set_center_reduce_normalizer_init():
    in_size = 4
    module = HyperEdgeSetCenterReduceNormalizer(n_features=in_size, update_limit=10)
    assert module.n_features == in_size
    assert module.updates[...] == 0
    assert module.mean.shape == (in_size,)
    assert module.var.shape == (in_size,)
    assert jnp.all(module.mean[...] == 0.0)
    assert jnp.all(module.var[...] == 1.0)


def test_center_reduce_module_validation():
    # Valid
    HyperEdgeSetCenterReduceNormalizer(n_features=1, update_limit=10)

    # Invalid
    with pytest.raises(ValueError, match="saturation_strategy must be None, 'hard' or 'soft'"):
        HyperEdgeSetCenterReduceNormalizer(n_features=1, update_limit=10, saturation_strategy="foo")


def test_center_reduce_normalizer_validation():
    struct = GraphStructure(hyper_edge_sets={"nodes": HyperEdgeSetStructure(port_list=[], feature_list=["a"])})

    # Valid
    CenterReduceNormalizer(struct, update_limit=10, saturation_strategy="hard", clip_min=-1.0, clip_max=1.0)
    CenterReduceNormalizer(struct, update_limit=10, saturation_strategy="soft")
    CenterReduceNormalizer(struct, update_limit=10)  # Default is None

    # Invalid strategy
    with pytest.raises(ValueError, match="saturation_strategy must be None, 'hard' or 'soft'"):
        CenterReduceNormalizer(struct, update_limit=10, saturation_strategy="invalid")

    # Missing clip for hard
    with pytest.raises(ValueError, match="clip_min and clip_max must be provided"):
        CenterReduceNormalizer(struct, update_limit=10, saturation_strategy="hard")

    # Invalid clip range
    with pytest.raises(ValueError, match="clip_min must be strictly less than clip_max"):
        CenterReduceNormalizer(struct, update_limit=10, saturation_strategy="hard", clip_min=1.0, clip_max=-1.0)


def test_hyper_edge_set_center_reduce_normalizer_update():
    in_size = 2
    module = HyperEdgeSetCenterReduceNormalizer(n_features=in_size, update_limit=3, beta_1=0.0, beta_2=0.0)

    # First update: with beta=0, it should take current mean/var
    x = jnp.array([[1.0, 10.0], [3.0, 30.0]])
    mask = jnp.ones((2, 1))
    _ = module(x, mask)

    assert module.updates[...] == 1
    # Mean of [1, 3] is 2, [10, 30] is 20
    assert jnp.allclose(module.mean[...], jnp.array([2.0, 20.0]))
    # Var of [1, 3] is 1, [10, 30] is 100
    assert jnp.allclose(module.var[...], jnp.array([1.0, 100.0]))

    # Multiple updates until limit
    for _ in range(5):
        module(x, mask)

    assert module.updates[...] == 3


def test_hyper_edge_set_center_reduce_normalizer_normalization_range():
    in_size = 1
    # Use beta=0 to have immediate convergence for testing
    module = HyperEdgeSetCenterReduceNormalizer(n_features=in_size, update_limit=100, beta_1=0.0, beta_2=0.0)

    # Train on a large sample to match empirical mean/std
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (10000, 1)) * 2.0 + 5.0
    mask = jnp.ones((10000, 1))

    # Single update is enough with beta=0
    module(data, mask)

    # In the module, bias correction is applied:
    # mean_hat = self.mean / (1 - beta_1**updates + epsilon)
    # var_hat = self.var / (1 - beta_2**updates + epsilon)
    # With beta=0 and updates=1, mean_hat = self.mean / (1 - 0 + epsilon) approx self.mean

    # Let's verify what the module actually uses for normalization
    emp_mean = module.mean[0] / (1 - module.beta_1 ** module.updates[0] + module.epsilon)
    emp_var = module.var[0] / (1 - module.beta_2 ** module.updates[0] + module.epsilon)
    emp_std = jnp.sqrt(emp_var)

    # Switch to inference mode to test normalization without updating statistics further
    module.use_running_average = True

    # Test normalization on these values
    test_data = jnp.array([[emp_mean], [emp_mean + emp_std], [emp_mean - emp_std]])
    norm_out = module(test_data, jnp.ones((3, 1)))

    # Mean should be 0.0
    assert jnp.allclose(norm_out[0, 0], 0.0, atol=1e-5)
    # +1 std should be 1.0
    assert jnp.allclose(norm_out[1, 0], 1.0, atol=1e-5)
    # -1 std should be -1.0
    assert jnp.allclose(norm_out[2, 0], -1.0, atol=1e-5)


def test_hyper_edge_set_center_reduce_normalizer_inference_mode():
    module = HyperEdgeSetCenterReduceNormalizer(n_features=1, update_limit=10, use_running_average=True)
    initial_mean = module.mean[...]

    data = jnp.array([[1.0], [2.0], [3.0]])
    module(data, jnp.ones((3, 1)))

    assert module.updates[...] == 0
    assert jnp.all(module.mean[...] == initial_mean)


def test_hyper_edge_set_center_reduce_normalizer_masking():
    module = HyperEdgeSetCenterReduceNormalizer(n_features=1, update_limit=10, beta_1=0.0, beta_2=0.0)

    # Only the first element is non-fictitious
    data = jnp.array([[100.0], [0.0]])
    mask = jnp.array([[1.0], [0.0]])

    module(data, mask)

    # Mean/Var should reflect only the first element
    assert jnp.allclose(module.mean, 100.0)
    assert jnp.allclose(module.var, 0.0)


def test_hyper_edge_set_center_reduce_normalizer_batch_shape():
    in_size = 2
    module = HyperEdgeSetCenterReduceNormalizer(n_features=in_size, update_limit=10)

    # Batch of 2 examples, each with 3 nodes
    data = jnp.ones((2, 3, 2))
    mask = jnp.ones((2, 3, 1))

    out = module(data, mask)
    assert out.shape == (2, 3, 2)
    assert module.updates[...] == 1


def test_center_reduce_normalizer_init():
    struct = GraphStructure(
        hyper_edge_sets={
            "nodes": HyperEdgeSetStructure(port_list=["id"], feature_list=["a", "b"]),
            "edges": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None),
        }
    )
    normalizer = CenterReduceNormalizer(struct, update_limit=10)
    assert "nodes" in normalizer.module_dict
    assert isinstance(normalizer.module_dict["nodes"], HyperEdgeSetCenterReduceNormalizer)
    assert normalizer.module_dict["edges"] is None


def test_center_reduce_normalizer_call():
    # Use jax_context and context_structure from the file setup
    struct = pb_batch.context_structure
    normalizer = CenterReduceNormalizer(struct, update_limit=10, return_metrics=True)

    # Call normalizer
    norm_graph, info = normalizer(graph=jax_context, step_with_metrics=True)

    assert isinstance(norm_graph, Graph)
    assert "input_graph" in info
    assert "output_graph" in info

    # Check that some values changed in nodes features
    for k in jax_context.hyper_edge_sets:
        if normalizer.module_dict.get(k) is not None:
            original = jax_context.hyper_edge_sets[k].feature_array
            normalized = norm_graph.hyper_edge_sets[k].feature_array
            if original is not None and original.shape[-2] > 0:
                assert not jnp.allclose(original, normalized)


def test_center_reduce_normalizer_multi_sets():
    struct = GraphStructure(
        hyper_edge_sets={
            "s1": HyperEdgeSetStructure(port_list=["p1"], feature_list=["f1"]),
            "s2": HyperEdgeSetStructure(port_list=["p2"], feature_list=["f2", "f3"]),
            "s3": HyperEdgeSetStructure(port_list=["p3"], feature_list=None),
        }
    )
    normalizer = CenterReduceNormalizer(struct, update_limit=10)

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
        true_shape=None,
        current_shape=None,
    )

    norm_graph, _ = normalizer(graph=graph)

    # Check that s1 and s2 are normalized
    assert not jnp.allclose(norm_graph.hyper_edge_sets["s1"].feature_array, h1.feature_array)
    assert not jnp.allclose(norm_graph.hyper_edge_sets["s2"].feature_array, h2.feature_array)
    # Check that s3 is unchanged
    assert norm_graph.hyper_edge_sets["s3"].feature_array is None


def test_center_reduce_normalizer_update_limit():
    struct = GraphStructure(hyper_edge_sets={"s1": HyperEdgeSetStructure(port_list=["p1"], feature_list=["f1"])})
    normalizer = CenterReduceNormalizer(struct, update_limit=2)

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


def test_center_reduce_saturation(caplog):
    feature_names = ["feat1"]
    hyper_edge_sets = {"nodes": HyperEdgeSetStructure(port_list=[], feature_list=feature_names)}
    structure = GraphStructure(hyper_edge_sets=hyper_edge_sets)

    # Enable saturation (hard)
    normalizer = CenterReduceNormalizer(
        in_structure=structure,
        update_limit=10,
        saturation_strategy="hard",
        clip_min=-1.0,
        clip_max=1.0,
    )

    # Fit with data (mean=0, std=1)
    data = jnp.array([-1.0, 1.0]).reshape(1, 2, 1)
    nodes = HyperEdgeSet(
        backend=JaxBackend(), port_dict=None, feature_array=data, feature_names=feature_names, non_fictitious=jnp.ones((1, 2))
    )
    graph = Graph.from_dict(backend=JaxBackend(), hyper_edge_set_dict={"nodes": nodes}, n_addresses=2)

    for _ in range(5):
        normalizer(graph=graph)

    normalizer.use_running_average = True

    # Test with outlier
    outlier_data = jnp.array([[[10.0]]])
    outlier_nodes = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=None,
        feature_array=outlier_data,
        feature_names=feature_names,
        non_fictitious=jnp.ones((1, 1)),
    )
    outlier_graph = Graph.from_dict(backend=JaxBackend(), hyper_edge_set_dict={"nodes": outlier_nodes}, n_addresses=1)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        gn, _ = normalizer(graph=outlier_graph)

    val = gn.hyper_edge_sets["nodes"].feature_array[0, 0, 0]
    assert val == 1.0
    assert "Normalization saturation occurred" in caplog.text


def test_center_reduce_soft_saturation(caplog):
    feature_names = ["feat1"]
    hyper_edge_sets = {"nodes": HyperEdgeSetStructure(port_list=[], feature_list=feature_names)}
    structure = GraphStructure(hyper_edge_sets=hyper_edge_sets)

    # Enable saturation (soft)
    normalizer = CenterReduceNormalizer(
        in_structure=structure,
        update_limit=10,
        saturation_strategy="soft",
        clip_min=-1.0,
        clip_max=1.0,
    )

    # Fit with data (mean=0, std=1)
    data = jnp.array([-1.0, 1.0]).reshape(1, 2, 1)
    nodes = HyperEdgeSet(
        backend=JaxBackend(), port_dict=None, feature_array=data, feature_names=feature_names, non_fictitious=jnp.ones((1, 2))
    )
    graph = Graph.from_dict(backend=JaxBackend(), hyper_edge_set_dict={"nodes": nodes}, n_addresses=2)

    for _ in range(5):
        normalizer(graph=graph)

    normalizer.use_running_average = True

    # Test with outlier
    outlier_data = jnp.array([[[2.0]]])
    outlier_nodes = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=None,
        feature_array=outlier_data,
        feature_names=feature_names,
        non_fictitious=jnp.ones((1, 1)),
    )
    outlier_graph = Graph.from_dict(backend=JaxBackend(), hyper_edge_set_dict={"nodes": outlier_nodes}, n_addresses=1)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        gn, _ = normalizer(graph=outlier_graph)

    val = gn.hyper_edge_sets["nodes"].feature_array[0, 0, 0]
    # tanh(2.0) is ~0.964
    assert val < 1.0
    assert "Normalization saturation occurred" not in caplog.text


@pytest.mark.parametrize(
    "return_metrics, step_with_metrics, expect_metrics",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_center_reduce_normalizer_return_metrics_switch(return_metrics, step_with_metrics, expect_metrics):
    """Quantile metrics are returned only when enabled at construction AND on a step with metrics."""
    struct = pb_batch.context_structure
    normalizer = CenterReduceNormalizer(struct, update_limit=10, return_metrics=return_metrics)

    _, metrics = normalizer(graph=jax_context, step_with_metrics=step_with_metrics)

    assert ("input_graph" in metrics) is expect_metrics
    assert ("output_graph" in metrics) is expect_metrics
