# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
"""Unit tests for :class:`GlobalAggregationMessagePassingFunction`.

Mirrors the structure of :mod:`test_message_fonction` (the
``LocalSumMessagePassingFunction`` suite) and :mod:`test_gatv2_message_passing`
so the three message functions are tested at parity. Adds tests specific to
the global-aggregation pattern:

- single ``value_mlp`` head (no per-(class, port) factoring);
- broadcast property: every real receiver gets the same global summary;
- corrected denominator ``sum(non_fictitious_addresses) + eps`` (the
  scientific-concern #1 resolution proposed in attention-backlog sec 3.2);
- permutation INVARIANCE (stronger than equivariance — the mean is
  symmetric in its arguments and the broadcast value does not depend on
  the order of addresses).
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from energnn.graph.jax import JaxGraph
from energnn.model.coupler.message_passing.message_passing_function import (
    GlobalAggregationMessagePassingFunction,
)
from energnn.problem.example import LinearSystemProblemLoader

np.random.seed(0)

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _make_default_gagg(out_size: int = 3, hidden_sizes=(4,), seed: int = 0):
    return GlobalAggregationMessagePassingFunction(
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


def _patch_identity_mlp(mf: GlobalAggregationMessagePassingFunction) -> None:
    """Replace the single Linear layer of `value_mlp` with an identity kernel.

    Lets tests assert analytical mean correctness independent of MLP randomness.
    Only works when the MLP has zero hidden layers (hidden_sizes=[]) and
    in_array_size == out_size.
    """
    linear = mf.value_mlp.sequential.layers[0]
    in_features = linear.in_features
    out_features = linear.out_features
    assert in_features == out_features, "identity patch requires square Linear"
    linear.kernel.value = jnp.eye(in_features, dtype=jnp.float32)
    if linear.bias is not None:
        linear.bias.value = jnp.zeros((out_features,), dtype=jnp.float32)


def test_value_mlp_initialised_with_correct_in_out_sizes():
    """``value_mlp`` is a single MLP (not a per-(class, port) tree) operating
    on coordinate vectors directly. Its input dim must match ``in_array_size``
    and its output dim must match ``out_size``.
    """
    mf = _make_default_gagg(out_size=5)
    last = mf.value_mlp.sequential.layers[-1]
    assert hasattr(last, "out_features")
    assert last.out_features == 5
    first = mf.value_mlp.sequential.layers[0]
    assert first.in_features == coordinates.shape[1]


def test_eps_default_is_one_e_minus_nine():
    """The ``eps`` numerical guard in the mean denominator defaults to 1e-9.
    It only matters in the degenerate case of zero non-fictitious addresses;
    in the normal regime the denominator equals ``sum(non_fictitious)`` exactly.
    """
    mf = _make_default_gagg()
    assert mf.eps == 1e-9


def test_output_shape_matches_n_addr_and_out_size():
    """Output is the global summary broadcast to every receiving address,
    so its shape must be ``(n_addr, out_size)`` regardless of input feature size.
    """
    mf = _make_default_gagg(out_size=5)
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    assert out.shape == (coordinates.shape[0], 5)


def test_output_is_finite():
    """Forward pass must not produce NaN or Inf on a standard small input."""
    mf = _make_default_gagg()
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_output_deterministic_under_repeated_calls():
    """With a fixed seed and unchanged input, two forward calls in the same
    process must produce identical output (no hidden randomness in the path)."""
    mf = _make_default_gagg(seed=7)
    out_a, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    out_b, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_close(out_a, out_b, atol=1e-7)


def test_output_broadcasts_same_value_across_real_receivers():
    """Every real (non-fictitious) receiver gets the exact same global summary —
    output rows must be byte-identical on the non-fictitious subset. This is
    the defining property of global aggregation."""
    mf = _make_default_gagg()
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    mask = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    real_rows = np.asarray(out)[mask]
    if real_rows.shape[0] < 2:
        return  # context has < 2 real addresses; nothing to compare
    max_row_diff = float(np.max(np.abs(real_rows - real_rows[0])))
    assert max_row_diff == 0.0, f"broadcast broken: max row diff {max_row_diff}"


def test_fictitious_receivers_get_zero_output():
    """Padding addresses must contribute zero to every downstream layer:
    rows of output corresponding to ``non_fictitious_addresses == 0`` must be
    exactly zero."""
    mf = _make_default_gagg()
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    mask = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    if mask.all():
        return  # no fictitious receivers in this fixture
    fictitious_rows = np.asarray(out)[~mask]
    assert float(np.max(np.abs(fictitious_rows))) == 0.0


def test_mean_correctness_with_patched_identity_mlp():
    """Patch ``value_mlp`` to identity (zero hidden, no bias, identity kernel)
    so ``value_mlp(coordinates) == coordinates``. The output is then the mean
    of coordinates over non-fictitious addresses, broadcast back. Asserts
    bit-exact equality with the analytical mean."""
    mf = _make_default_gagg(out_size=coordinates.shape[1], hidden_sizes=())
    # Recreate with use_bias=False so identity kernel + zero bias = identity.
    mf = GlobalAggregationMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=coordinates.shape[1],
        use_bias=False,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=0,
    )
    _patch_identity_mlp(mf)
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    expected_mean = jnp.sum(coordinates * jnp.expand_dims(jax_context.non_fictitious_addresses, -1), axis=0) / (
        jnp.sum(jax_context.non_fictitious_addresses) + mf.eps
    )
    mask = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    first_real_idx = int(np.argmax(mask))
    chex.assert_trees_all_close(out[first_real_idx], expected_mean, atol=1e-6)


def test_corrected_denominator_uses_non_fictitious_count():
    """The specification denominator ``|A_x|`` includes fictitious padding
    and dilutes the mean. The implementation must divide by
    ``sum(non_fictitious_addresses) + eps`` instead.

    Test: synthetic graph with 3 real / 7 fictitious addresses, all coords = 1
    after the patched identity MLP. Expected mean per dim = 1.0 (sum 3 / 3),
    NOT 0.3 (sum 3 / 10)."""
    n_addr = 10
    n_real = 3
    fake_mask = jnp.array([1.0] * n_real + [0.0] * (n_addr - n_real), dtype=jnp.float32)
    fake_ctx = JaxGraph(
        hyper_edge_sets=jax_context.hyper_edge_sets,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=fake_mask,
    )
    fake_coords = jnp.ones((n_addr, coordinates.shape[1]), dtype=jnp.float32)
    mf = GlobalAggregationMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=coordinates.shape[1],
        use_bias=False,
        final_activation=None,
        outer_activation=nnx.identity,
        seed=0,
    )
    _patch_identity_mlp(mf)
    out, _ = mf(graph=fake_ctx, coordinates=fake_coords, get_info=False)
    # First real row: mean over 3 real coords (all 1.0) -> 1.0 per dim.
    chex.assert_trees_all_close(out[0], jnp.ones((coordinates.shape[1],), dtype=jnp.float32), atol=1e-6)
    # Not 0.3 (which would be sum(3) / |A_x|=10).
    assert float(out[0, 0]) > 0.9, f"denominator wrong: got {float(out[0, 0])}, expected ~1.0"


def test_permutation_invariance_of_aggregated_mean():
    """Mean is symmetric in its arguments, so permuting addresses must leave
    the (broadcast) mean value unchanged. Stronger than the permutation
    equivariance enjoyed by LocalSum / GATv2 (where output rows permute with
    the address indices); here the output value itself is invariant."""
    mf = _make_default_gagg(seed=11)
    out_orig, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
    # Permute addresses with a fixed random order.
    rng = np.random.default_rng(7)
    n_addr = coordinates.shape[0]
    perm = jnp.array(rng.permutation(n_addr))
    perm_coords = coordinates[perm]
    perm_mask = jax_context.non_fictitious_addresses[perm]
    perm_ctx = JaxGraph(
        hyper_edge_sets=jax_context.hyper_edge_sets,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=perm_mask,
    )
    out_perm, _ = mf(graph=perm_ctx, coordinates=perm_coords, get_info=False)
    # Both outputs broadcast the same mean to every real receiver. Pick a real
    # row from each and assert byte-identical equality.
    mask_orig = np.asarray(jax_context.non_fictitious_addresses).astype(bool)
    mask_perm = np.asarray(perm_mask).astype(bool)
    if not mask_orig.any() or not mask_perm.any():
        return
    first_orig = np.asarray(out_orig)[mask_orig][0]
    first_perm = np.asarray(out_perm)[mask_perm][0]
    chex.assert_trees_all_close(first_orig, first_perm, atol=1e-6)


def test_gradient_flows_through_value_mlp():
    """A loss computed on the output must produce non-zero gradients on every
    ``value_mlp`` parameter — otherwise training would be impossible."""
    mf = _make_default_gagg(seed=3)

    def loss(m):
        out, _ = m(graph=jax_context, coordinates=coordinates, get_info=False)
        return jnp.mean(out ** 2)

    val, grads = nnx.value_and_grad(loss)(mf)
    _, params, _ = nnx.split(grads, nnx.Param, ...)
    grad_leaves = [g for g in jax.tree_util.tree_leaves(params) if hasattr(g, "size")]
    total_norm = float(sum(jnp.sum(g ** 2) for g in grad_leaves) ** 0.5)
    assert total_norm > 0.0, "gradient L2 norm is zero — value_mlp not differentiable"


def test_vmap_jit_safety_after_build():
    """Composition of ``nnx.jit`` and ``nnx.vmap`` must work on a batched
    coordinates input; this is exactly how ``RecurrentCoupler`` calls the
    message function inside a batched training step."""
    mf = _make_default_gagg(seed=5)

    def step(m, g, c):
        out, _ = m(graph=g, coordinates=c, get_info=False)
        return out

    batched = nnx.jit(nnx.vmap(step, in_axes=(None, None, 0), out_axes=0))
    out_batched = batched(mf, jax_context, coordinates_batch)
    assert out_batched.shape == (coordinates_batch.shape[0], coordinates.shape[0], mf.out_size)
    assert bool(jnp.all(jnp.isfinite(out_batched)))


def test_seed_xor_rngs_validation():
    """Constructor must reject the ``(seed=..., rngs=...)`` ambiguous pairing.
    Passing both is a user mistake that would silently shadow one source of
    randomness with the other."""
    rngs = nnx.Rngs(0)
    try:
        GlobalAggregationMessagePassingFunction(
            in_graph_structure=pb_loader.context_structure,
            in_array_size=coordinates.shape[1],
            hidden_sizes=[4],
            out_size=3,
            seed=1,
            rngs=rngs,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError when both seed and rngs are provided.")
