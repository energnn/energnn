# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import numpy as np
import jax
import jax.numpy as jnp

from energnn.normalizer.normalization_function.identity_function import IdentityFunction


def test_construction_and_aux_methods_return_expected_values():
    fn = IdentityFunction()
    arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    aux = fn.init_aux(arr)
    assert aux is None

    aux2 = fn.update_aux(arr, aux)
    assert aux2 is None

    params = fn.compute_params(arr, aux)
    # compute_params returns an empty array
    assert isinstance(params, jnp.ndarray)
    assert params.shape == (0,)
    assert params.size == 0


def test_apply_masks_values_correctly_basic():
    fn = IdentityFunction()
    array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([1.0, 0.0])  # second row masked
    out = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask)

    expected = np.array([[1.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-8)


def test_apply_broadcasting_mask_variants():
    fn = IdentityFunction()
    array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask1 = jnp.array([1.0, 0.0])
    mask2 = jnp.array([[1.0], [0.0]])
    mask_bool = jnp.array([True, False])

    out1 = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask1)
    out2 = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask2)
    out3 = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask_bool)

    # Expected: mask applied per-line -> second row zeroed entirely
    expected = np.array([[1.0, 0.0], [3.0, 0.0]])
    expected1 = np.array([[1.0, 2.0], [0.0, 0.0]])

    np.testing.assert_allclose(np.array(out1), expected, rtol=1e-6)
    np.testing.assert_allclose(np.array(out2), expected1, rtol=1e-6)
    np.testing.assert_allclose(np.array(out3), expected, rtol=1e-6)

    # Additional sanity: shapes preserved and dtype consistent
    assert out1.shape == array.shape
    assert out2.shape == array.shape
    assert out3.shape == array.shape


def test_apply_with_integer_input_upcasts_and_values_preserved():
    fn = IdentityFunction()
    array_int = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    mask = jnp.array([1.0, 1.0])
    out = fn.apply(params=jnp.array([]), array=array_int, non_fictitious=mask)

    expected = np.array(array_int.astype(jnp.float32)) * np.array(mask)[:, None]
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6)


def test_apply_inverse_equals_apply():
    fn = IdentityFunction()
    array = jnp.array([[7.0, -1.5], [0.0, 3.3]])
    mask = jnp.array([1.0, 0.0])
    out_apply = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask)
    out_inv = fn.apply_inverse(params=jnp.array([]), array=array, non_fictitious=mask)

    np.testing.assert_allclose(np.array(out_apply), np.array(out_inv), rtol=1e-6)


def test_gradient_inverse_returns_ones_and_is_masked():
    fn = IdentityFunction()
    array = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    mask = jnp.array([1.0, 0.0])
    grad = fn.gradient_inverse(params=jnp.array([]), array=array, non_fictitious=mask)

    expected = np.ones_like(np.array(array)) * np.array(mask)[None, :]
    np.testing.assert_allclose(np.array(grad), expected, rtol=1e-6)


def test_jit_compatibility_apply_and_gradient_inverse():
    fn = IdentityFunction()
    array = jnp.array([[1.2, -3.4], [5.6, 7.8]])
    mask = jnp.array([1.0, 0.0])

    jit_apply = jax.jit(fn.apply)
    jit_grad = jax.jit(fn.gradient_inverse)

    out_nonjit = fn.apply(params=jnp.array([]), array=array, non_fictitious=mask)
    out_jit = jit_apply(params=jnp.array([]), array=array, non_fictitious=mask)
    np.testing.assert_allclose(np.array(out_nonjit), np.array(out_jit), rtol=1e-6)

    grad_nonjit = fn.gradient_inverse(params=jnp.array([]), array=array, non_fictitious=mask)
    grad_jit = jit_grad(params=jnp.array([]), array=array, non_fictitious=mask)
    np.testing.assert_allclose(np.array(grad_nonjit), np.array(grad_jit), rtol=1e-6)


def test_vmap_compatibility_apply_batching():
    fn = IdentityFunction()
    # build a small batch (B, N, D)
    batch = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    masks = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # vectorize apply across batch dimension
    batched_apply = jax.vmap(fn.apply, in_axes=(None, 0, 0))

    # call with positional arguments (not keywords) to match in_axes
    out = batched_apply(jnp.array([]), batch, masks)

    # manual per-example expected
    expected0 = np.array(batch[0]) * np.array(masks[0])[None, :]
    expected1 = np.array(batch[1]) * np.array(masks[1])[None, :]
    np.testing.assert_allclose(np.array(out[0]), expected0, rtol=1e-6)
    np.testing.assert_allclose(np.array(out[1]), expected1, rtol=1e-6)
