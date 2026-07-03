#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
"""Unit tests for :class:`GATv2MessagePassingFunction`.

Mirrors the structure of :mod:`test_message_fonction` (the
``LocalSumMessagePassingFunction`` test suite) so the two message
functions are tested at parity. Adds tests specific to attention:
the softmax normaliser, the ``score_uses_receiver`` flag, and
permutation equivariance under the attention-weighted aggregation.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from energnn.graph.jax import JaxGraph, JaxHyperEdgeSet
from energnn.model.coupler.message_passing.message_passing_function import (
    GATv2MessagePassingFunction,
)
from energnn.problem.example import LinearSystemProblemLoader

np.random.seed(0)

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _make_default_gatv2(out_size: int = 3, hidden_sizes=(4,), score_uses_receiver=None, seed: int = 0):
    kwargs: dict = dict(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=list(hidden_sizes),
        activation=nnx.leaky_relu,
        out_size=out_size,
        use_bias=True,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=seed,
    )
    if score_uses_receiver is not None:
        kwargs["score_uses_receiver"] = score_uses_receiver
    return GATv2MessagePassingFunction(**kwargs)


def test_score_and_value_trees_initialised_from_structure():
    mf = _make_default_gatv2()
    expected_keys = set(pb_loader.context_structure.hyper_edge_sets.keys())
    assert set(mf.score_mlp_tree.keys()) == expected_keys
    assert set(mf.value_mlp_tree.keys()) == expected_keys
    for ek in expected_keys:
        edge_struct = pb_loader.context_structure.hyper_edge_sets[ek]
        expected_ports = set(edge_struct.port_list)
        assert set(mf.score_mlp_tree[ek].keys()) == expected_ports
        assert set(mf.value_mlp_tree[ek].keys()) == expected_ports
        for pk in expected_ports:
            assert callable(mf.score_mlp_tree[ek][pk])
            assert callable(mf.value_mlp_tree[ek][pk])


def test_score_mlp_outputs_scalar_value_mlp_outputs_vector():
    out_size = 5
    mf = _make_default_gatv2(out_size=out_size)
    # Score MLPs have out_features == 1
    for ek in mf.score_mlp_tree.keys():
        for pk in mf.score_mlp_tree[ek].keys():
            # When final_activation is None the Sequential's last entry is a Linear.
            last = mf.score_mlp_tree[ek][pk].sequential.layers[-1]
            assert hasattr(last, "out_features"), "expected last layer to be Linear"
            assert last.out_features == 1, "score MLP must emit a scalar logit"
    # Value MLPs have out_features == out_size
    for ek in mf.value_mlp_tree.keys():
        for pk in mf.value_mlp_tree[ek].keys():
            last = mf.value_mlp_tree[ek][pk].sequential.layers[-1]
            assert hasattr(last, "out_features")
            assert last.out_features == out_size


def test_score_in_size_grows_when_score_uses_receiver_true():
    mf_false = _make_default_gatv2(score_uses_receiver=False)
    mf_true = _make_default_gatv2(score_uses_receiver=True)
    for ek in mf_false.score_mlp_tree.keys():
        for pk in mf_false.score_mlp_tree[ek].keys():
            in_size_false = mf_false.score_mlp_tree[ek][pk].sequential.layers[0].in_features
            in_size_true = mf_true.score_mlp_tree[ek][pk].sequential.layers[0].in_features
            assert in_size_true == in_size_false + coordinates.shape[1], (
                f"score_uses_receiver=True must extend score MLP input by in_array_size; "
                f"got {in_size_false} -> {in_size_true} for ({ek}, {pk})"
            )
    # Value MLP input size is unchanged
    for ek in mf_false.value_mlp_tree.keys():
        for pk in mf_false.value_mlp_tree[ek].keys():
            v_false = mf_false.value_mlp_tree[ek][pk].sequential.layers[0].in_features
            v_true = mf_true.value_mlp_tree[ek][pk].sequential.layers[0].in_features
            assert v_true == v_false


def test_output_shape_and_dtype():
    out_size = 5
    mf = _make_default_gatv2(out_size=out_size)
    out, info = mf(graph=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (coordinates.shape[0], out_size)
    assert info == {}


def test_output_is_finite():
    mf = _make_default_gatv2()
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    assert bool(jnp.all(jnp.isfinite(out))), "GATv2 must not emit NaN/inf on a healthy input"


def test_deterministic_with_seed():
    mf1 = _make_default_gatv2(seed=7)
    mf2 = _make_default_gatv2(seed=7)
    out1, _ = mf1(graph=jax_context, coordinates=coordinates, get_info=False)
    out2, _ = mf2(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_close(out1, out2, atol=1e-7)


def test_score_uses_receiver_default_is_true_matching_gatv2_paper():
    """The constructor default of `score_uses_receiver` must be True
    (Brody et al. 2022 -- `[h_a || h_e]` concatenation). Approche 1 (default score_uses_receiver=True per Brody et al. 2022)."""
    mf_default = _make_default_gatv2(seed=11)
    mf_explicit_true = _make_default_gatv2(score_uses_receiver=True, seed=11)
    out_default, _ = mf_default(graph=jax_context, coordinates=coordinates, get_info=False)
    out_explicit, _ = mf_explicit_true(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_close(out_default, out_explicit, atol=0.0)


def test_score_uses_receiver_true_differs_from_false():
    """Sanity: the two settings must produce materially different outputs
    (otherwise the flag is dead code)."""
    mf_false = _make_default_gatv2(seed=13, score_uses_receiver=False)
    mf_true = _make_default_gatv2(seed=13, score_uses_receiver=True)
    out_false, _ = mf_false(graph=jax_context, coordinates=coordinates, get_info=False)
    out_true, _ = mf_true(graph=jax_context, coordinates=coordinates, get_info=False)
    diff = float(jnp.max(jnp.abs(out_false - out_true)))
    assert diff > 1e-6, "score_uses_receiver flag must change the output materially"


def test_segment_max_subtraction_handles_large_scores():
    """Numerical stability: with scores in the range that would overflow
    naive ``exp(s)`` (s ~ 100 -> exp = 2.7e43, beyond float32 max ~ 3.4e38),
    the segment-max subtraction must keep the output finite. Approche 1 numerical-stability case."""
    mf = _make_default_gatv2(seed=17)
    # Drive scores high by inflating the input coordinates. The MLPs are
    # leaky_relu + Linear, so a large-norm input produces a large-magnitude
    # logit that would overflow without max-subtraction.
    big_coordinates = coordinates * 1e3
    out, _ = mf(graph=jax_context, coordinates=big_coordinates, get_info=False)
    assert jnp.all(jnp.isfinite(out)), "segment-max subtraction must prevent overflow on large scores"


def test_non_fictitious_masking_zeros_padded_contribution():
    """Fictitious objects must not contribute to either numerator or denominator."""
    n_addr = 4
    d = coordinates.shape[1]
    coords = jnp.array(np.random.uniform(size=(n_addr, d)).astype(np.float32))
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    # All-real edge
    edge_real = JaxHyperEdgeSet(
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.array([1.0, 1.0, 1.0]),
    )
    # Same edge structure but middle object is fictitious
    edge_fict = JaxHyperEdgeSet(
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.array([1.0, 0.0, 1.0]),
    )
    g_real = JaxGraph(
        hyper_edge_sets={"line": edge_real},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    g_fict = JaxGraph(
        hyper_edge_sets={"line": edge_fict},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    # Build a smaller graph structure for the message function: line only with 2 ports, no features.
    from energnn.graph import GraphStructure, HyperEdgeSetStructure

    struct = GraphStructure(
        hyper_edge_sets={
            "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None),
        }
    )
    mf = GATv2MessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=d,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=d,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=20,
    )
    out_real, _ = mf(graph=g_real, coordinates=coords, get_info=False)
    out_fict, _ = mf(graph=g_fict, coordinates=coords, get_info=False)
    # With the middle object fictitious, addresses touched only by that object
    # (here address 2 via port "to") receive zero contribution. address 2 should
    # be zero in the fictitious graph.
    out_fict_np = np.array(out_fict)
    assert np.allclose(
        out_fict_np[2], 0.0, atol=1e-6
    ), "address only touched by a fictitious object should have zero attention output"
    # And the real-edge output at address 2 should be non-trivially different.
    assert not np.allclose(np.array(out_real)[2], out_fict_np[2], atol=1e-6)


def test_softmax_normaliser_uniform_with_equal_scores():
    """With deterministic patched MLPs giving constant scores, attention weights
    should be uniform across the per-receiver neighbour set, so the output
    equals the mean of the neighbour values rather than their sum.
    """
    n_addr = 3
    d = coordinates.shape[1]
    coords = jnp.array(np.random.uniform(size=(n_addr, d)).astype(np.float32))
    addr_from = jnp.array([0, 0, 0])  # all three edges send to receiver 1 via port "to"
    addr_to = jnp.array([1, 1, 1])
    edge = JaxHyperEdgeSet(
        port_dict={"from": addr_from, "to": addr_to},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.array([1.0, 1.0, 1.0]),
    )
    from energnn.graph import GraphStructure, HyperEdgeSetStructure

    struct = GraphStructure(
        hyper_edge_sets={
            "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None),
        }
    )
    g = JaxGraph(
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    mf = GATv2MessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=d,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=d,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=30,
    )

    # Patch all score MLPs to return the same constant (so softmax weights are uniform 1/N).
    class _ConstantScore:
        def __call__(self, x):
            return jnp.zeros((x.shape[0], 1), dtype=jnp.float32)

    # Patch value MLPs to identity-shaped constants per port: "from" gives [1,0,...], "to" gives [0,1,...].
    class _ConstantVec:
        def __init__(self, vec):
            self.vec = jnp.asarray(vec, dtype=jnp.float32)

        def __call__(self, x):
            return jnp.tile(self.vec[None, :], (x.shape[0], 1))

    for ek in list(mf.score_mlp_tree.keys()):
        for pk in list(mf.score_mlp_tree[ek].keys()):
            mf.score_mlp_tree[ek][pk] = _ConstantScore()
    from_vec = np.zeros(d, dtype=np.float32)
    from_vec[0] = 1.0
    to_vec = np.zeros(d, dtype=np.float32)
    to_vec[1] = 1.0
    mf.value_mlp_tree["line"]["from"] = _ConstantVec(from_vec)
    mf.value_mlp_tree["line"]["to"] = _ConstantVec(to_vec)

    out, _ = mf(graph=g, coordinates=coords, get_info=False)
    out_np = np.array(out)
    # All three edges contribute to receiver 1, all via port "to" with the same constant value.
    # The denominator is sum of 3 equal exp(0)=1 -> 3. Numerator is 3 * to_vec. Ratio = to_vec.
    expected_receiver_1 = to_vec
    np.testing.assert_allclose(out_np[1], expected_receiver_1, rtol=1e-5, atol=1e-5)
    # Receiver 0 sees three edges via port "from" with from_vec; output is from_vec.
    expected_receiver_0 = from_vec
    np.testing.assert_allclose(out_np[0], expected_receiver_0, rtol=1e-5, atol=1e-5)
    # Receiver 2 has no incoming neighbours -> output is zero (num=0, den=eps, ratio=0).
    np.testing.assert_allclose(out_np[2], np.zeros(d, dtype=np.float32), atol=1e-5)


def test_permutation_equivariance_under_address_permutation():
    """Permuting addresses and the corresponding coords must permute output the same way."""
    mf = _make_default_gatv2(seed=40)
    n_addr = coordinates.shape[0]
    rng = np.random.default_rng(0)
    perm = np.arange(n_addr)
    rng.shuffle(perm)
    inv = np.zeros(n_addr, dtype=int)
    for i, p in enumerate(perm):
        inv[p] = i

    out_orig, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)

    # Permute coords and remap port indices through inv.
    from copy import deepcopy

    ctx_perm = deepcopy(jax_context)
    for k, hes in ctx_perm.hyper_edge_sets.items():
        if hes.port_dict is not None:
            for port_name, port_arr in hes.port_dict.items():
                hes.port_dict[port_name] = jnp.array(inv[np.asarray(port_arr).astype(int)])
    coords_perm = coordinates[perm]
    out_perm, _ = mf(graph=ctx_perm, coordinates=coords_perm, get_info=False)
    # The output for the permuted graph, when un-permuted, must match the original.
    np.testing.assert_allclose(np.array(out_orig), np.array(out_perm[inv]), rtol=1e-6, atol=1e-6)


def test_gradient_flows_through_both_score_and_value_trees():
    """Both MLP trees must receive non-zero gradients (no dead branches)."""
    mf = _make_default_gatv2(seed=50)
    graphdef, params, rest = nnx.split(mf, nnx.Param, ...)

    def loss_fn(p):
        m = nnx.merge(graphdef, p, rest)
        out, _ = m(graph=jax_context, coordinates=coordinates, get_info=False)
        return jnp.sum(out**2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    nonzero = sum(int(jnp.any(jnp.abs(g) > 0)) for g in leaves)
    assert nonzero == len(leaves), (
        f"{len(leaves) - nonzero}/{len(leaves)} param leaves have zero gradient " "(dead branch in either score or value tree)"
    )


def test_vmap_jit_safety_after_build():
    mf = _make_default_gatv2(seed=60, hidden_sizes=(2,), out_size=4)
    apply_vmap = jax.vmap(
        lambda g, c, gi: mf(graph=g, coordinates=c, get_info=gi),
        in_axes=(0, 0, None),
        out_axes=0,
    )
    out1, info1 = apply_vmap(jax_context_batch, coordinates_batch, False)
    out2, info2 = jax.jit(apply_vmap)(jax_context_batch, coordinates_batch, False)
    chex.assert_trees_all_close(out1, out2, atol=1e-6)
    assert info1 == {} and info2 == {}
    assert np.array(out1).shape[0] == coordinates_batch.shape[0]


def test_empty_graph_returns_zeros():
    g = JaxGraph(
        hyper_edge_sets={},
        non_fictitious_addresses=jnp.ones((5,)),
        true_shape=None,
        current_shape=None,
    )
    mf = _make_default_gatv2(seed=70, out_size=3)
    out, _ = mf(graph=g, coordinates=jnp.zeros((5, coordinates.shape[1])), get_info=False)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(np.array(out), np.zeros((5, 3)), atol=1e-6)


def test_addresses_out_of_bounds_handling():
    """Out-of-bounds indices in port_dict are dropped silently by gather/scatter_add."""
    d = coordinates.shape[1]
    coords = jnp.array(np.random.uniform(size=(2, d)).astype(np.float32))
    edge = JaxHyperEdgeSet(
        port_dict={"from": jnp.array([0, 10]), "to": jnp.array([0, 1])},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.array([1.0, 1.0]),
    )
    from energnn.graph import GraphStructure, HyperEdgeSetStructure

    struct = GraphStructure(
        hyper_edge_sets={
            "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None),
        }
    )
    g = JaxGraph(
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((2,)),
        true_shape=None,
        current_shape=None,
    )
    mf = GATv2MessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=d,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=4,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=80,
    )
    out, _ = mf(graph=g, coordinates=coords, get_info=False)
    assert out.shape == (coords.shape[0], 4)
    assert bool(jnp.all(jnp.isfinite(out)))
