#
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
"""Unit tests for :class:`PerformerMessagePassingFunction`.

Mirrors the structure of :mod:`test_multi_head_qkv_message_passing` so the
attention message functions are tested at parity. Adds tests specific to
the linear-attention all-to-all form: kernel-trick parity (Form A naive
vs Form B used in the implementation), and the absence of graph
topology in the aggregation.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from energnn.graph.jax import JaxGraph
from energnn.model.coupler.message_passing.message_passing_function import (
    PerformerMessagePassingFunction,
)
from energnn.problem.example import LinearSystemProblemLoader

np.random.seed(0)

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _make_default_performer(
    out_size: int = 3,
    hidden_sizes=(4,),
    d_qk: int = 8,
    scale_scores=None,
    seed: int = 0,
):
    kwargs: dict = dict(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=list(hidden_sizes),
        d_qk=d_qk,
        activation=nnx.leaky_relu,
        out_size=out_size,
        use_bias=True,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=seed,
    )
    if scale_scores is not None:
        kwargs["scale_scores"] = scale_scores
    return PerformerMessagePassingFunction(**kwargs)


def test_qkv_mlps_initialised_with_correct_dims():
    """Constructor builds three per-address MLPs: query_mlp, key_mlp, value_mlp.

    Per backlog sec 3.4, Q and K are :math:`\\mathbb{R}^{d_{QK}}` and V is
    :math:`\\mathbb{R}^{d_V}` (which we map to ``out_size``). All three
    operate on coordinates (in_array_size -> {d_qk, d_qk, out_size}).
    """
    d_qk, out_size = 8, 3
    mf = _make_default_performer(out_size=out_size, d_qk=d_qk)
    assert isinstance(mf.query_mlp, nnx.Module)
    assert isinstance(mf.key_mlp, nnx.Module)
    assert isinstance(mf.value_mlp, nnx.Module)
    q = mf.query_mlp(coordinates)
    k = mf.key_mlp(coordinates)
    v = mf.value_mlp(coordinates)
    assert q.shape == (10, d_qk)
    assert k.shape == (10, d_qk)
    assert v.shape == (10, out_size)


def test_output_shape_and_dtype():
    """Forward output is (n_addr, out_size) and finite float32."""
    mf = _make_default_performer(out_size=3)
    out, info = mf(graph=jax_context, coordinates=coordinates)
    assert out.shape == (coordinates.shape[0], 3)
    assert out.dtype == jnp.float32
    assert info == {}


def test_output_is_finite():
    """No NaN / Inf in the forward output under default config."""
    mf = _make_default_performer()
    out, _ = mf(graph=jax_context, coordinates=coordinates)
    assert jnp.all(jnp.isfinite(out))


def test_deterministic_with_seed():
    """Same seed produces identical output bit-for-bit."""
    mf1 = _make_default_performer(seed=42)
    mf2 = _make_default_performer(seed=42)
    out1, _ = mf1(graph=jax_context, coordinates=coordinates)
    out2, _ = mf2(graph=jax_context, coordinates=coordinates)
    chex.assert_trees_all_close(out1, out2, atol=0.0, rtol=0.0)


def test_seed_xor_rngs_validation():
    """Passing both seed and rngs raises; otherwise default seed=0."""
    rngs = nnx.Rngs(7)
    with pytest.raises(ValueError):
        PerformerMessagePassingFunction(
            in_graph_structure=pb_loader.context_structure,
            in_array_size=7,
            hidden_sizes=[4],
            d_qk=8,
            out_size=3,
            seed=1,
            rngs=rngs,
        )


def test_default_scale_scores_is_true_matching_vaswani_2017():
    """Default deviates from the unscaled specification to follow the
    Vaswani 2017 stability convention (variance of the bilinear score grows
    with d_qk if not scaled)."""
    mf = _make_default_performer()
    assert mf.scale_scores is True


def test_scale_scores_true_differs_from_false():
    """Toggling scale_scores changes the output (the flag is not dead)."""
    mf_on = _make_default_performer(scale_scores=True, seed=5)
    mf_off = _make_default_performer(scale_scores=False, seed=5)
    out_on, _ = mf_on(graph=jax_context, coordinates=coordinates)
    out_off, _ = mf_off(graph=jax_context, coordinates=coordinates)
    assert not jnp.allclose(out_on, out_off, atol=1e-6)


def test_scale_scores_divides_by_sqrt_d_qk():
    """Switching scaling on divides the raw aggregator by sqrt(d_qk).

    Because the aggregator is linear in the score and the score is linear
    in the scaling factor, output_off = sqrt(d_qk) * output_on exactly.
    Identity outer_activation makes this directly observable.
    """
    d_qk = 16
    mf_on = _make_default_performer(d_qk=d_qk, scale_scores=True, seed=5)
    mf_off = _make_default_performer(d_qk=d_qk, scale_scores=False, seed=5)
    out_on, _ = mf_on(graph=jax_context, coordinates=coordinates)
    out_off, _ = mf_off(graph=jax_context, coordinates=coordinates)
    expected_off = jnp.sqrt(jnp.float32(d_qk)) * out_on
    chex.assert_trees_all_close(out_off, expected_off, atol=1e-5, rtol=1e-5)


def test_kernel_trick_form_matches_naive_form():
    """Form A (naive O(n^2) per-receiver sum) and Form B (kernel-trick,
    used in the implementation) produce numerically identical outputs.

    This is the core parity test of the Performer spec: the kernel-trick
    rephrasing in backlog sec 3.4 is mathematically equivalent to the
    naive per-receiver sum, and any divergence here would indicate a
    transposition or einsum bug.
    """
    mf = _make_default_performer(d_qk=8, out_size=3, seed=11)
    out_B, _ = mf(graph=jax_context, coordinates=coordinates)  # impl form B

    # Form A: per-receiver naive sum, exactly as spec defines.
    q = mf.query_mlp(coordinates)
    k = mf.key_mlp(coordinates)
    v = mf.value_mlp(coordinates)
    mask = jnp.expand_dims(jax_context.non_fictitious_addresses, -1)
    k_m = k * mask
    v_m = v * mask
    scale = 1.0 / jnp.sqrt(jnp.float32(mf.d_qk))

    n_addr = coordinates.shape[0]
    out_A = jnp.zeros((n_addr, mf.out_size), dtype=jnp.float32)
    for a in range(n_addr):
        scores = jnp.einsum("nd,d->n", k_m, q[a])  # (n_addr,)
        out_a = jnp.einsum("nd,n->d", v_m, scores) * scale
        out_A = out_A.at[a].set(out_a)
    chex.assert_trees_all_close(out_A, out_B, atol=1e-4, rtol=1e-4)


def test_non_fictitious_masking_zeros_padded_contribution():
    """Perturbing coordinates of fictitious addresses by a large factor
    does not change the output on real receivers.

    Justification: K and V are multiplied by the non_fictitious mask
    before the outer-product accumulation, so fictitious entries
    contribute zero to the sum. Q is read at each receiver address but a
    REAL receiver's Q depends only on its own (real) coordinate.
    """
    mf = _make_default_performer(seed=11)
    mask_np = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    out_ref, _ = mf(graph=jax_context, coordinates=coordinates)
    coords_pert = coordinates.at[~mask_np].set(coordinates[~mask_np] * 1e6)
    out_pert, _ = mf(graph=jax_context, coordinates=coords_pert)
    diff_real = jnp.max(jnp.abs(out_pert[mask_np] - out_ref[mask_np]))
    assert float(diff_real) < 1e-3


def test_permutation_equivariance_under_address_permutation():
    """Permuting addresses (coordinates + non_fictitious mask) permutes
    the output the same way: output[perm[i]] == output_perm[i].

    Performer ignores graph topology (port_dict), so equivariance is
    tested only on the coord+mask permutation. The new graph is built
    with the original hyper_edge_sets (untouched) and a permuted
    non_fictitious_addresses.
    """
    mf = _make_default_performer(seed=13)
    out, _ = mf(graph=jax_context, coordinates=coordinates)
    rng = np.random.default_rng(0)
    perm = rng.permutation(coordinates.shape[0])
    coords_perm = coordinates[perm]
    ctx_perm = JaxGraph(
        hyper_edge_sets=jax_context.hyper_edge_sets,
        non_fictitious_addresses=jax_context.non_fictitious_addresses[perm],
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    out_perm, _ = mf(graph=ctx_perm, coordinates=coords_perm)
    expected = out[perm]
    chex.assert_trees_all_close(out_perm, expected, atol=1e-5, rtol=1e-5)


def test_gradient_flows_through_q_k_v_mlps():
    """Gradient of a scalar loss w.r.t. coordinates is non-zero on real
    addresses (Q, K, V branches all contribute)."""
    mf = _make_default_performer(seed=17)

    def loss_fn(coords):
        out, _ = mf(graph=jax_context, coordinates=coords)
        return jnp.sum(out * out)

    grad = jax.grad(loss_fn)(coordinates)
    assert grad.shape == coordinates.shape
    assert jnp.all(jnp.isfinite(grad))
    mask = jax_context.non_fictitious_addresses[:, None]
    assert float(jnp.sum(jnp.abs(grad) * mask)) > 0.0


def test_vmap_jit_safety_after_build():
    """vmap + jit composition over a batched context produces correct
    shape and remains finite."""
    mf = _make_default_performer(seed=19, out_size=3)

    def f(coords, graph):
        out, _ = mf(graph=graph, coordinates=coords)
        return out

    f_vmap = nnx.jit(nnx.vmap(f, in_axes=(0, 0), out_axes=0))
    out = f_vmap(coordinates_batch, jax_context_batch)
    assert out.shape == (4, 10, 3)
    assert jnp.all(jnp.isfinite(out))


def test_empty_graph_returns_zeros_after_outer_activation():
    """All-fictitious input produces zero output (identity outer_activation).

    With every address fictitious, K_masked = V_masked = 0, so the outer
    product M is zero and any Q multiplication is zero. The Performer
    output is zero on every address.
    """
    mf = _make_default_performer(seed=23, out_size=3)
    fict_ctx = JaxGraph(
        hyper_edge_sets=jax_context.hyper_edge_sets,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    out, _ = mf(graph=fict_ctx, coordinates=coordinates)
    chex.assert_trees_all_close(out, jnp.zeros_like(out), atol=1e-6)


def test_output_varies_across_receivers_via_q():
    """Distinguishes Performer from GlobalAggregation: Performer outputs
    differ across receivers because each receiver has its own Q_a, while
    GlobalAggregation broadcasts the same mean to every receiver.

    Two real receivers should produce different output rows when Q_a
    differs (which it does when h_a differs).
    """
    mf = _make_default_performer(seed=29, out_size=3)
    out, _ = mf(graph=jax_context, coordinates=coordinates)
    mask_np = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    real_idx = np.where(mask_np)[0]
    assert real_idx.size >= 2, "test needs at least 2 real addresses"
    diff = float(jnp.max(jnp.abs(out[real_idx[0]] - out[real_idx[1]])))
    assert diff > 1e-4, "outputs identical across receivers - Q dependence missing"
