# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import numpy as np
import jax
import jax.numpy as jnp

from energnn.normalizer.normalization_function.cdf_pw_linear_function import (
    CDFPWLinearFunction,
    get_proba_quantiles,
    merge_equal_quantiles,
    forward,
    inverse,
)


def test_init_aux_and_update_aux_and_compute_params_shapes():
    rng = np.random.RandomState(0)
    # create two small batches with 2 features
    a = jnp.array(rng.normal(size=(4, 2)))
    b = jnp.array(rng.normal(size=(3, 2)))

    nf = CDFPWLinearFunction(n_breakpoints=4)
    aux = nf.init_aux(a)
    assert isinstance(aux, list) and len(aux) == 0

    aux = nf.update_aux(a, aux)
    assert len(aux) == 1 and aux[0].shape == (4, 2)
    aux = nf.update_aux(b, aux)
    assert len(aux) == 2

    params = nf.compute_params(None, aux)
    # params has shape (2, n_breakpoints + 1, n_features)
    assert isinstance(params, jnp.ndarray)
    assert params.shape[0] == 2
    assert params.shape[1] == 4 + 1
    assert params.shape[2] == 2


def test_get_proba_quantiles_basic_and_nanquantile_equivalence():
    # deterministic sorted data per feature
    x = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    n_breakpoints = 4
    p, q = get_proba_quantiles(x, n_breakpoints)

    # p should be vector [0, 1/4, 2/4, 3/4, 1] expanded to shape (n+1, n_features)
    expected_p = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # p returned is (n+1, n_features), so per-column must equal expected_p
    np.testing.assert_allclose(np.array(p[:, 0]), expected_p, rtol=0, atol=0)
    np.testing.assert_allclose(np.array(p[:, 1]), expected_p, rtol=0, atol=0)

    # q should be nanquantile(x, p, axis=0)
    # check with numpy for tolerance
    numpy_q0 = np.nanquantile(np.array(x[:, 0]), expected_p)
    numpy_q1 = np.nanquantile(np.array(x[:, 1]), expected_p)
    np.testing.assert_allclose(np.array(q[:, 0]), numpy_q0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(q[:, 1]), numpy_q1, rtol=1e-6, atol=1e-6)


def test_merge_equal_quantiles_constant_column():
    # simulate a column with equal quantiles (constant data)
    p = jnp.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    q = jnp.array([[5.0], [5.0], [5.0], [5.0], [5.0]])  # all identical
    p_merged, q_merged = merge_equal_quantiles(p, q)

    # q_merged should equal original q (still constant)
    np.testing.assert_allclose(np.array(q_merged[:, 0]), np.array([5.0] * 5), rtol=0, atol=0)
    # p_merged should be averaged across identical q; since all q identical,
    # unique value has probability equal to average p -> repeated
    avg_p = float(np.mean(np.array([0.0, 0.25, 0.5, 0.75, 1.0])))
    # After merge implementation unique_p.at[inverse].add(p) / count then p = unique_p.at[inverse].get()
    # thus p_merged will equal repeated avg_p
    np.testing.assert_allclose(np.array(p_merged[:, 0]), np.array([avg_p] * 5), rtol=1e-6, atol=1e-6)


def test_forward_inverse_identity_on_grid_and_extrapolation():
    # simple xp/fp monotone mapping
    xp = jnp.array([1.0, 2.0, 3.0])
    fp = jnp.array([-1.0, 0.0, 1.0])

    # test a grid inside [1,3]
    xs = jnp.array([1.0, 1.5, 2.0, 2.5, 3.0])
    fxs = forward(xs, xp, fp)
    inv = inverse(fxs, fp, xp)

    np.testing.assert_allclose(np.array(inv), np.array(xs), rtol=1e-6, atol=1e-6)

    # extrapolate left and right
    x_left = jnp.array(-1.0)
    x_right = jnp.array(5.0)
    f_left = forward(x_left, xp, fp)
    f_right = forward(x_right, xp, fp)
    # invert back
    inv_left = inverse(f_left, fp, xp)
    inv_right = inverse(f_right, fp, xp)

    np.testing.assert_allclose(np.array(inv_left), np.array(x_left), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(inv_right), np.array(x_right), rtol=1e-5, atol=1e-5)


def test_compute_params_and_apply_and_apply_inverse_restore_masked_values():
    # Build a small dataset and compute params
    # Two feature columns
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0]])
    nf = CDFPWLinearFunction(n_breakpoints=4)
    aux = nf.init_aux(a)
    aux = nf.update_aux(a, aux)
    aux = nf.update_aux(b, aux)
    params = nf.compute_params(None, aux)
    # params shape OK
    assert params.shape == (2, 5, 2)

    # Build an array to normalize and mask (3 rows, 2 features)
    array = jnp.array([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    # mask: second row fictitious -> mark with 0
    # Use mask shaped (n,1) to avoid JAX rank-promotion issues in some configs
    mask = jnp.array([1.0, 0.0, 1.0]).reshape((3, 1))

    normalized = nf.apply(params, array, mask)
    # masked row should be zero
    np.testing.assert_allclose(np.array(normalized[1]), np.zeros((2,)), rtol=1e-6, atol=1e-6)
    # other rows should be finite numbers
    assert np.all(np.isfinite(np.array(normalized[[0, 2],])))

    # apply inverse -> restored values for non-fictitious rows approximately equal original
    restored = nf.apply_inverse(params, normalized, mask)
    # second row remains zero *should be (masked)*
    np.testing.assert_allclose(np.array(restored[1]), np.zeros((2,)), rtol=1e-6, atol=1e-6)
    # rows 0 and 2 should be close to original
    np.testing.assert_allclose(np.array(restored[0]), np.array(array[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(restored[2]), np.array(array[2]), rtol=1e-5, atol=1e-5)


def test_jit_compatibility_forward_inverse_and_apply():
    # Build simple xp/fp mapping and test jax.jit doesn't crash and works numerically
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-1.0, 0.0, 1.0])
    xs = jnp.linspace(0.0, 2.0, 5)

    jf = jax.jit(forward)
    ji = jax.jit(inverse)
    fxs = jf(xs, xp, fp)
    inv = ji(fxs, fp, xp)

    np.testing.assert_allclose(np.array(inv), np.array(xs), rtol=1e-6, atol=1e-6)

    # JIT compute_params via small aux - ensure no exception (use the class)
    nf = CDFPWLinearFunction(n_breakpoints=3)
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    aux = nf.init_aux(a)
    aux = nf.update_aux(a, aux)

    # compute_params is not jitted in implementation, but apply/inverse should be
    params = nf.compute_params(None, aux)
    apply_jit = jax.jit(nf.apply)
    mask = jnp.ones((2, 1))
    out = apply_jit(params, a, mask)
    assert out.shape == a.shape


def test_gradient_inverse_simple_constant_slope():
    """
    For a simple bijective mapping xp=[1,2,3], fp=[-1,0,1],
    inverse(f) = forward(f, fp, xp) has local slope (xp[i+1]-xp[i])/(fp[i+1]-fp[i]) = 1.0 everywhere.
    Thus gradient_inverse should return 1.0 for any input f (including extrapolation).
    """

    # build params for a single feature
    xp = jnp.array([1.0, 2.0, 3.0])  # original x breakpoints
    fp = jnp.array([-1.0, 0.0, 1.0])  # mapped f breakpoints
    # params shape expected by function: (2, n_breakpoints, n_features)
    params = jnp.stack([xp[:, None], fp[:, None]], axis=0)  # shape (2, 3, 1)

    nf = CDFPWLinearFunction(n_breakpoints=2)  # n_breakpoints not used here since we pass params manually

    # test array of normalized values (f) including points inside and outside the fp range
    f_values = jnp.array([[-2.0], [-0.5], [0.0], [0.5], [2.0]])  # shape (5,1)
    mask = jnp.ones((5, 1))

    grad = nf.gradient_inverse(params, f_values, mask)
    # All slopes should be 1.0 for this simple mapping
    expected = jnp.ones_like(f_values)
    np.testing.assert_allclose(np.array(grad), np.array(expected), rtol=1e-6, atol=1e-6)


def test_gradient_inverse_masking_and_jit_compatibility():
    """
    Ensure gradient_inverse honors the non_fictitious mask (masked rows -> zero),
    and that the function works under jax.jit.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-1.0, 0.0, 1.0])
    params = jnp.stack([xp[:, None], fp[:, None]], axis=0)  # shape (2,3,1)

    nf = CDFPWLinearFunction(n_breakpoints=2)

    # Build f-values and a mask that marks the middle row as fictitious (0)
    f_values = jnp.array([[0.0], [0.5], [-0.5]])  # shape (3,1)
    mask = jnp.array([[1.0], [0.0], [1.0]])

    # non-jit
    grad = nf.gradient_inverse(params, f_values, mask)
    # the middle row must be zeroed out by mask
    assert np.allclose(np.array(grad[1]), np.zeros((1,)), atol=1e-8)

    # the other rows should be finite positive slopes
    assert np.all(np.isfinite(np.array(grad[[0, 2],])))

    # test JIT compatibility
    grad_jit = jax.jit(nf.gradient_inverse)(params, f_values, mask)
    np.testing.assert_allclose(np.array(grad_jit), np.array(grad), rtol=1e-6, atol=1e-8)
