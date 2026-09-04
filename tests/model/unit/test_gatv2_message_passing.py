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

from energnn.model.coupler.message_passing.message_passing_function import GATv2MessagePassingFunction
from energnn.problem.example import LinearSystemProblemLoader

np.random.seed(0)

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _make_gatv2(n_heads=1, out_size=8, seed=0, **kwargs):
    return GATv2MessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        n_heads=n_heads,
        out_size=out_size,
        seed=seed,
        **kwargs,
    )


def test_mlp_tree_fuses_scores_and_values():
    n_heads, out_size = 2, 8
    mf = _make_gatv2(n_heads=n_heads, out_size=out_size, seed=0)
    assert set(mf.mlp_tree.keys()) == set(pb_loader.context_structure.hyper_edge_sets.keys())
    for edge_key in mf.mlp_tree:
        for mlp in mf.mlp_tree[edge_key].values():
            # Fused output is n_heads scores plus out_size value channels.
            assert mlp.sequential.layers[-1].out_features == out_size + n_heads


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_forward_shape_and_finite(n_heads):
    mf = _make_gatv2(n_heads=n_heads, out_size=8, seed=0)
    out, info = mf(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    assert out.shape == (10, 8)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert info == {}


def test_out_size_must_be_divisible_by_n_heads():
    with pytest.raises(ValueError):
        _make_gatv2(n_heads=3, out_size=8, seed=0)


def test_deterministic_with_seed():
    out_a, _ = _make_gatv2(n_heads=2, seed=5)(graph=jax_context, coordinates=coordinates)
    out_b, _ = _make_gatv2(n_heads=2, seed=5)(graph=jax_context, coordinates=coordinates)
    chex.assert_trees_all_close(out_a, out_b, atol=0.0)


def test_attention_weights_sum_to_one():
    # With every score equal and every value equal to a constant, the softmax
    # weights sum to one, so each receiving address returns the constant
    # regardless of its number of incoming neighbors.
    mf = _make_gatv2(n_heads=2, out_size=4, outer_activation=nnx.identity, seed=0)
    constant = 0.37

    class _ConstantMLP:
        def __call__(self, x):
            n_edges = x.shape[0]
            score = jnp.zeros((n_edges, mf.n_heads))
            value = jnp.full((n_edges, mf.out_size), constant)
            return jnp.concatenate([score, value], axis=-1)

    for edge_key in list(mf.mlp_tree.keys()):
        for port_key in list(mf.mlp_tree[edge_key].keys()):
            mf.mlp_tree[edge_key][port_key] = _ConstantMLP()

    out, _ = mf(graph=jax_context, coordinates=coordinates)
    out_flat = out.reshape(-1)
    is_zero = jnp.abs(out_flat) < 1e-5
    is_constant = jnp.abs(out_flat - constant) < 1e-5
    # Every entry is either the constant (address received) or zero (address had no neighbor).
    assert bool(jnp.all(is_zero | is_constant))
    assert bool(jnp.any(is_constant))


def test_fictitious_receivers_are_zero():
    mf = _make_gatv2(n_heads=2, out_size=4, outer_activation=nnx.identity, seed=1)
    out, _ = mf(graph=jax_context, coordinates=coordinates)
    fictitious = 1.0 - jax_context.non_fictitious_addresses
    fictitious_rows = out * fictitious[:, None]
    chex.assert_trees_all_close(fictitious_rows, jnp.zeros_like(fictitious_rows), atol=1e-6)


def test_permutation_equivariance():
    # Relabeling the addresses by a random permutation permutes the output the
    # same way. New address j holds old address perm[j]; a port to old address a
    # is remapped to its new label perm_inv[a].
    mf = _make_gatv2(n_heads=2, out_size=4, seed=2)
    out, _ = mf(graph=jax_context, coordinates=coordinates)

    rng = np.random.RandomState(0)
    perm = rng.permutation(10)
    perm_inv = jnp.array(np.argsort(perm))
    perm = jnp.array(perm)

    permuted_edges = {}
    for edge_key, edge in jax_context.hyper_edge_sets.items():
        permuted_ports = {name: perm_inv[arr.astype(int)].astype(arr.dtype) for name, arr in edge.port_dict.items()}
        permuted_edges[edge_key] = type(edge)(
            backend=edge._backend,
            port_dict=permuted_ports,
            feature_array=edge.feature_array,
            feature_names=edge.feature_names,
            non_fictitious=edge.non_fictitious,
        )
    permuted_graph = type(jax_context)(
        backend=jax_context._backend,
        hyper_edge_sets=permuted_edges,
        non_fictitious_addresses=jax_context.non_fictitious_addresses[perm],
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    out_permuted, _ = mf(graph=permuted_graph, coordinates=coordinates[perm])
    chex.assert_trees_all_close(out_permuted, out[perm], atol=1e-5)


def test_segment_max_handles_large_scores():
    mf = _make_gatv2(n_heads=2, out_size=4, outer_activation=nnx.identity, seed=0)
    out, _ = mf(graph=jax_context, coordinates=coordinates * 1.0e3)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_multiple_heads_are_distinct():
    single, _ = _make_gatv2(n_heads=1, out_size=8, seed=7)(graph=jax_context, coordinates=coordinates)
    multi, _ = _make_gatv2(n_heads=4, out_size=8, seed=7)(graph=jax_context, coordinates=coordinates)
    assert float(jnp.max(jnp.abs(single - multi))) > 1e-6


def test_vmap_jit_safety():
    mf = _make_gatv2(n_heads=2, out_size=8, seed=1)

    def forward(message_function, graph, coords):
        return message_function(graph=graph, coordinates=coords, step_with_metrics=False)[0]

    batched = nnx.jit(nnx.vmap(forward, in_axes=(None, 0, 0), out_axes=0))
    out = batched(mf, jax_context_batch, coordinates_batch)
    assert out.shape == (4, 10, 8)
