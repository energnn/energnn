#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from energnn.normalizer.normalization_function.center_reduce_function import (
    CenterReduceFunction,
    forward,
    inverse,
)


def test_init_aux_returns_empty_list():
    cr = CenterReduceFunction(epsilon=1e-8)
    aux = cr.init_aux(jnp.array([[1.0, 2.0]]))
    assert isinstance(aux, list)
    assert len(aux) == 0


def test_update_aux_appends_arrays():
    cr = CenterReduceFunction()
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0]])
    aux = cr.init_aux(a)
    aux = cr.update_aux(a, aux)
    aux = cr.update_aux(b, aux)
    assert len(aux) == 2
    np.testing.assert_allclose(np.array(aux[0]), np.array(a))
    np.testing.assert_allclose(np.array(aux[1]), np.array(b))


def test_compute_params_mean_std_matches_numpy():
    cr = CenterReduceFunction()
    # three rows, two features
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, -1.0]])
    aux = [a, b]
    params = cr.compute_params(None, aux)
    # params shape (2, n_features): [mean, std]
    assert params.shape == (2, 2)
    dataset = np.concatenate([np.array(x) for x in aux], axis=0)
    expected_mean = np.nanmean(dataset, axis=0)
    expected_std = np.nanstd(dataset, axis=0)
    np.testing.assert_allclose(np.array(params[0]), expected_mean)
    np.testing.assert_allclose(np.array(params[1]), expected_std)


def test_compute_params_handles_nans():
    cr = CenterReduceFunction()
    a = jnp.array([[1.0, jnp.nan], [3.0, 4.0]])
    b = jnp.array([[jnp.nan, 6.0]])
    aux = [a, b]
    params = cr.compute_params(None, aux)
    dataset = np.concatenate([np.array(x) for x in aux], axis=0)
    expected_mean = np.nanmean(dataset, axis=0)
    expected_std = np.nanstd(dataset, axis=0)
    np.testing.assert_allclose(np.array(params[0]), expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(params[1]), expected_std, rtol=1e-6, atol=1e-6)


def test_forward_inverse_composition_simple():
    # simple per-feature test
    x = jnp.array([1.5, -2.0, 0.3])
    mean = jnp.array([1.0, 0.0, 0.0])
    std = jnp.array([0.5, 2.0, 1.0])
    eps = 1e-8
    y = forward(x, mean, std, eps)
    x_rec = inverse(y, mean, std, eps)
    np.testing.assert_allclose(np.array(x_rec), np.array(x), rtol=1e-6, atol=1e-6)


def test_apply_and_apply_inverse_restore_masked_values():
    cr = CenterReduceFunction(epsilon=1e-8)
    # build dataset: 4 objects, 2 features
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0]])
    aux = [a, b]
    params = cr.compute_params(None, aux)

    # Test array to normalize (same number of features)
    array = jnp.array([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    # mask: mark the second row as fictitious (0), others valid (1)
    mask = jnp.array([1.0, 0.0, 1.0])[:, None]

    normed = cr.apply(params, array, mask)
    # masked entries must be zeroed where mask == 0
    assert normed.shape == array.shape
    assert np.allclose(np.array(normed[1]), np.zeros((2,)))  # masked line is zeros

    # inverse should restore original values on non-masked entries
    recon = cr.apply_inverse(params, normed, mask)
    # masked row should remain zeroed
    assert np.allclose(np.array(recon[1]), np.zeros((2,)))
    # unmasked rows equal corresponding original rows
    np.testing.assert_allclose(np.array(recon[0]), np.array(array[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(recon[2]), np.array(array[2]), rtol=1e-6, atol=1e-6)


def test_apply_handles_zero_std_with_epsilon():
    # feature 0 is constant -> std = 0
    cr = CenterReduceFunction(epsilon=1e-6)
    a = jnp.array([[2.0, 1.0], [2.0, 3.0]])  # feature 0 constant (2.0)
    aux = [a]
    params = cr.compute_params(None, aux)
    mean = np.array(params[0])
    std = np.array(params[1])
    # std[0] should be zero
    assert np.isclose(std[0], 0.0) or np.all(std[0] == 0.0)  # per-feature
    # apply and inverse should not produce NaN or inf
    array = jnp.array([[2.0, 1.0], [2.0, 3.0]])
    mask = jnp.array([1.0, 1.0])
    normed = cr.apply(params, array, mask)
    assert not jnp.isnan(normed).any()
    recon = cr.apply_inverse(params, normed, mask)
    np.testing.assert_allclose(np.array(recon), np.array(array), rtol=1e-6, atol=1e-6)


def test_gradient_inverse_equals_std_plus_epsilon():
    cr = CenterReduceFunction(epsilon=1e-7)
    a = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    aux = [a]
    params = cr.compute_params(None, aux)
    std = np.array(params[1])  # shape (n_features,)

    # Construct arbitrary normalized array (same shape as inputs)
    normalized = jnp.array([[0.1, -0.2], [0.3, 0.4], [0.0, 1.0]])
    mask = jnp.array([1.0, 0.0, 1.0])[:, None]  # second row masked out

    grad = cr.gradient_inverse(params, normalized, mask)
    # gradient should equal (std + eps) broadcasted across rows, and zeroed where mask == 0
    expected_per_feature = std + cr.epsilon
    # expand for check: shape (n_rows, n_features)
    expected = np.tile(expected_per_feature[None, :], (normalized.shape[0], 1))
    expected[1, :] = 0.0  # because mask[1] == 0
    np.testing.assert_allclose(np.array(grad), expected, rtol=1e-6, atol=1e-8)

def test_gradient_inverse_matches_expected_std_plus_epsilon():
    # For CenterReduceFunction inverse(x) = x * (std + eps) + mean
    # so d/dx inverse = std + eps

    # Build simple data: two features
    a = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cr = CenterReduceFunction(epsilon=1e-8)
    aux = cr.init_aux(a)
    aux = cr.update_aux(a, aux)
    params = cr.compute_params(None, aux)
    mean = np.array(params[0])
    std = np.array(params[1])

    # Choose arbitrary normalized array values (same shape as rows)
    arr = jnp.zeros((3, 2))
    mask = jnp.array([1.0, 1.0, 1.0]).reshape((3, 1))
    grad_inv = cr.gradient_inverse(params, arr, mask)
    # grad_inv should equal std + eps for each row (broadcasted)
    expected = (std + 1e-8)[None, :]
    np.testing.assert_allclose(np.array(grad_inv), np.repeat(expected, 3, axis=0), rtol=1e-6, atol=1e-8)


def test_shapes_and_broadcasting():
    cr = CenterReduceFunction()
    # aux with two rows, three features
    a = jnp.array([[1.0, 2.0, 3.0]])
    b = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    params = cr.compute_params(None, [a, b])
    # input with zero rows should keep shape
    empty = jnp.empty((0, 3))
    mask_empty = jnp.empty((0,))[:, None]
    normed_empty = cr.apply(params, empty, mask_empty)
    assert normed_empty.shape == (0, 3)

    # broadcasting: mask shape (n,) should broadcast on last axis
    array = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = jnp.array([1.0, 0.0])[:, None]
    normed = cr.apply(params, array, mask)
    assert normed.shape == array.shape
    assert np.allclose(np.array(normed[1]), np.zeros((3,)))


def test_jit_and_vmap_compatibility():
    cr = CenterReduceFunction(epsilon=1e-8)
    # build simple aux and params
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    params = cr.compute_params(None, [a])
    # input batch of 3 rows
    array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([1.0, 1.0])

    # jitted apply
    jitted_apply = jax.jit(cr.apply)
    normed_jit = jitted_apply(params, array, mask)
    normed = cr.apply(params, array, mask)
    np.testing.assert_allclose(np.array(normed_jit), np.array(normed), rtol=1e-6, atol=1e-8)

    # vmap on a batch dimension: create batch of different arrays (stack along leading axis)
    arrays = jnp.stack([array, array + 1.0], axis=0)  # shape (2, n, f)
    masks = jnp.stack([mask, mask], axis=0)
    # vmapped apply: we need a vmappable wrapper since apply signature is (params, array, mask)
    vmapped = jax.vmap(lambda prm, arr, m: cr.apply(prm, arr, m), in_axes=(None, 0, 0))
    out = vmapped(params, arrays, masks)
    assert out.shape == arrays.shape


def test_compute_params_empty_aux_raises():
    cr = CenterReduceFunction()
    with pytest.raises((ValueError, TypeError, jax.errors.ConcretizationTypeError, Exception)):
        cr.compute_params(None, [])


def test_center_reduce_numeric_mean_std_and_inverse():
    a = jnp.array([[1.0, 2.0],
                   [3.0, 4.0],
                   [5.0, 6.0]], dtype=jnp.float32)

    cr = CenterReduceFunction(epsilon=1e-8)

    aux = [a]

    params = cr.compute_params(None, aux)

    mask = jnp.ones((a.shape[0], 1), dtype=jnp.float32)

    normed = cr.apply(params, a, mask)

    mean_norm = jnp.nanmean(normed, axis=0)
    std_norm = jnp.nanstd(normed, axis=0)

    # compute expected stats from original data (numpy for reference)
    np_a = np.array(a)
    orig_mean = np.mean(np_a, axis=0)
    orig_std = np.std(np_a, axis=0)

    # expected std after forward: orig_std / (orig_std + epsilon)
    expected_std_norm = orig_std / (orig_std + cr.epsilon)

    # assert means ~ 0
    np.testing.assert_allclose(np.array(mean_norm), np.zeros_like(orig_mean), rtol=1e-6, atol=1e-6)

    # assert stds close to theoretical value
    np.testing.assert_allclose(np.array(std_norm), expected_std_norm, rtol=1e-6, atol=1e-6)

    # now test inverse: apply_inverse should restore original (on non-fictitious rows)
    denorm = cr.apply_inverse(params, normed, mask)
    np.testing.assert_allclose(np.array(denorm), np_a, rtol=1e-6, atol=1e-6)


def test_center_reduce_with_fictitious_mask_preserves_zero_rows():
    # dataset (3 objets, 2 features)
    a = jnp.array([[10.0, -1.0],
                   [0.5, 0.5],
                   [3.0, 4.0]], dtype=jnp.float32)

    cr = CenterReduceFunction(epsilon=1e-8)
    aux = [a]
    params = cr.compute_params(None, aux)

    # mask: mark the second row as fictitious (0), others valid (1)
    mask = jnp.array([[1.0], [0.0], [1.0]], dtype=jnp.float32)

    normed = cr.apply(params, a, mask)

    # The masked row should be zeros after apply (because we multiply by mask)
    np.testing.assert_allclose(np.array(normed[1]), np.zeros((a.shape[1],)), atol=1e-6)

    # After inverse, masked row should remain zero (inverse(0) * 0 == 0)
    denorm = cr.apply_inverse(params, normed, mask)
    np.testing.assert_allclose(np.array(denorm[1]), np.zeros((a.shape[1],)), atol=1e-6)

    # Non-masked rows should be restored to their original values
    np.testing.assert_allclose(np.array(denorm[0]), np.array(a[0]), atol=1e-6)
    np.testing.assert_allclose(np.array(denorm[2]), np.array(a[2]), atol=1e-6)