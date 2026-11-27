#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from energnn.model.coupler.message_passing.message_passing_function import GlobalAggregationMessagePassingFunction
from energnn.problem.example import LinearSystemProblemLoader

np.random.seed(0)

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))

AGGREGATIONS = ["mean", "sum", "max", "min", "attention"]


def _with_address_mask(non_fictitious_addresses):
    """Returns the test context graph with another mask of non-fictitious addresses."""
    return type(jax_context)(
        backend=jax_context._backend,
        hyper_edge_sets=jax_context.hyper_edge_sets,
        non_fictitious_addresses=non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )


def _make_global(aggregation="mean", n_heads=1, out_size=8, seed=0, **kwargs):
    return GlobalAggregationMessagePassingFunction(
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        aggregation=aggregation,
        n_heads=n_heads,
        out_size=out_size,
        seed=seed,
        **kwargs,
    )


@pytest.mark.parametrize("aggregation, n_heads", [("mean", 1), ("max", 1), ("attention", 2)])
def test_mlp_out_size(aggregation, n_heads):
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=8)
    # With attention, the MLP also outputs one score per head.
    expected = 8 + n_heads if aggregation == "attention" else 8
    assert mf.mlp.sequential.layers[-1].out_features == expected


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_forward_shape_and_finite(aggregation):
    n_heads = 2 if aggregation == "attention" else 1
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=8)
    out, info = mf(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    assert out.shape == (10, 8)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert info == {}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"aggregation": "median"}, "Unknown aggregation"),
        ({"aggregation": "mean", "n_heads": 2}, "only used with"),
        ({"aggregation": "attention", "n_heads": 0}, "at least 1"),
        ({"aggregation": "attention", "n_heads": 3}, "divisible"),
        ({"seed": None}, "Either seed or rngs"),
        ({"rngs": nnx.Rngs(0)}, "Seed must be None"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_global(**kwargs)


@pytest.mark.parametrize("init", [{"seed": 5}, {"seed": None, "rngs": nnx.Rngs(5)}])
def test_deterministic_with_seed(init):
    out_a, _ = _make_global(aggregation="attention", n_heads=2, seed=5)(graph=jax_context, coordinates=coordinates)
    out_b, _ = _make_global(aggregation="attention", n_heads=2, **init)(graph=jax_context, coordinates=coordinates)
    chex.assert_trees_all_close(out_a, out_b, atol=1e-6)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "max", "min"])
def test_matches_reference(aggregation):
    mf = _make_global(aggregation=aggregation, outer_activation=nnx.identity, seed=3)
    out, _ = mf(graph=jax_context, coordinates=coordinates)

    mask = np.array(jax_context.non_fictitious_addresses) > 0
    assert 0 < mask.sum() < mask.size
    values = np.array(mf.mlp(coordinates))[mask]
    reducer = {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min}[aggregation]
    expected = reducer(values, axis=0)
    # Every non-fictitious address receives the same message, fictitious ones receive zero.
    np.testing.assert_allclose(np.array(out)[mask], np.broadcast_to(expected, (mask.sum(), 8)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(out)[~mask], 0.0, atol=1e-6)


def test_attention_matches_reference():
    n_heads, out_size = 2, 4
    mf = _make_global(aggregation="attention", n_heads=n_heads, out_size=out_size, outer_activation=nnx.identity, seed=4)
    out, _ = mf(graph=jax_context, coordinates=coordinates)

    mask = np.array(jax_context.non_fictitious_addresses) > 0
    output = np.array(mf.mlp(coordinates), dtype=np.float64)[mask]
    scores = output[:, :n_heads]
    values = output[:, n_heads:].reshape(-1, n_heads, out_size // n_heads)
    weights = np.exp(scores - scores.max(axis=0))
    weights = weights / weights.sum(axis=0)
    # Softmax over the non-fictitious addresses, separately for each head, heads concatenated.
    expected = np.einsum("nh,nhv->hv", weights, values).reshape(out_size)
    np.testing.assert_allclose(np.array(out)[mask], np.broadcast_to(expected, (mask.sum(), out_size)), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_fictitious_addresses_are_ignored(aggregation):
    # Changing the coordinates of fictitious addresses does not change the output.
    n_heads = 2 if aggregation == "attention" else 1
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=4, seed=1)
    out, _ = mf(graph=jax_context, coordinates=coordinates)
    fictitious = (1.0 - jax_context.non_fictitious_addresses)[:, None]
    assert float(jnp.sum(fictitious)) > 0
    out_noisy, _ = mf(graph=jax_context, coordinates=coordinates + 1.0e3 * fictitious)
    chex.assert_trees_all_close(out, out_noisy, atol=1e-6)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_permutation_equivariance(aggregation):
    n_heads = 2 if aggregation == "attention" else 1
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=4, seed=2)
    out, _ = mf(graph=jax_context, coordinates=coordinates)

    perm = jnp.array(np.random.RandomState(0).permutation(10))
    permuted_graph = _with_address_mask(jax_context.non_fictitious_addresses[perm])
    out_permuted, _ = mf(graph=permuted_graph, coordinates=coordinates[perm])
    chex.assert_trees_all_close(out_permuted, out[perm], atol=1e-5)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
@pytest.mark.parametrize("case", ["large_coordinates", "all_fictitious", "no_address"])
def test_extreme_inputs_are_finite(aggregation, case):
    # Large coordinates, a graph where every address is fictitious and a graph without addresses
    # give finite outputs and gradients; without non-fictitious address, the output is zero.
    n_heads = 2 if aggregation == "attention" else 1
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=4, outer_activation=nnx.identity, seed=0)
    graph, coords = {
        "large_coordinates": (jax_context, coordinates * 1.0e3),
        "all_fictitious": (_with_address_mask(jnp.zeros_like(jax_context.non_fictitious_addresses)), coordinates),
        "no_address": (_with_address_mask(jnp.zeros((0,))), coordinates[:0]),
    }[case]
    graphdef, params = nnx.split(mf)

    def loss(params, graph, coords):
        return jnp.sum(nnx.merge(graphdef, params)(graph=graph, coordinates=coords)[0] ** 2)

    out, _ = mf(graph=graph, coordinates=coords)
    grads = jax.grad(loss)(params, graph, coords)
    assert out.shape == (coords.shape[0], 4)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
    if case != "large_coordinates":
        chex.assert_trees_all_close(out, jnp.zeros_like(out))


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_vmap_jit_safety(aggregation):
    n_heads = 2 if aggregation == "attention" else 1
    mf = _make_global(aggregation=aggregation, n_heads=n_heads, out_size=8, seed=1)

    def forward(message_function, graph, coords):
        return message_function(graph=graph, coordinates=coords, step_with_metrics=False)[0]

    batched = nnx.jit(nnx.vmap(forward, in_axes=(None, 0, 0), out_axes=0))
    out = batched(mf, jax_context_batch, coordinates_batch)
    assert out.shape == (4, 10, 8)
