#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph.jax import JaxGraph, JaxEdge, JaxGraphShape
from energnn.normalizer import Postprocessor, Preprocessor
from energnn.normalizer.normalization_function import CDFPWLinearFunction, CenterReduceFunction, IdentityFunction
from tests.utils import TestProblemLoader

n = 10
pb_loader = TestProblemLoader(
    dataset_size=8,
    n_batch=4,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n, "feature_list": ["e"]},
        "edge": {"n_obj": n, "feature_list": ["f"]},
    },
    n_addr=n,
    shuffle=True,
)


# Negative tests: not fitted -> raises
def test_preprocessor_raises_if_not_fitted():
    pre = Preprocessor(f=IdentityFunction())
    empty_jax_graph = JaxGraph(edges={}, non_fictitious_addresses=jnp.array([]), true_shape=None, current_shape=None)

    # single methods
    with pytest.raises(RuntimeError):
        pre.preprocess(empty_jax_graph)
    with pytest.raises(RuntimeError):
        pre.preprocess_batch(empty_jax_graph)
    with pytest.raises(RuntimeError):
        pre.preprocess_inverse(empty_jax_graph)
    with pytest.raises(RuntimeError):
        pre.preprocess_inverse_batch(empty_jax_graph)


def test_postprocessor_raises_if_not_fitted():
    post = Postprocessor(f=IdentityFunction())
    empty_jax_graph = JaxGraph(edges={}, non_fictitious_addresses=jnp.array([]), true_shape=None, current_shape=None)

    with pytest.raises(RuntimeError):
        post.postprocess(empty_jax_graph)
    with pytest.raises(RuntimeError):
        post.postprocess_batch(empty_jax_graph)
    # For preconditioning methods both out_graph and grad_graph should be valid JaxGraph objects:
    with pytest.raises(RuntimeError):
        post.precondition_gradient(empty_jax_graph, empty_jax_graph)
    with pytest.raises(RuntimeError):
        post.precondition_gradient_batch(empty_jax_graph, empty_jax_graph)


# Pickle roundtrip tests
def test_preprocessor_and_postprocessor_pickle_roundtrip(tmp_path):
    pre = Preprocessor(f=CenterReduceFunction())
    pre._fitted = True
    pre.params = {"node": jnp.array([[0.0, 1.0], [0.0, 2.0]])}  # dummy
    pfile = tmp_path / "pre.pkl"
    pre.to_pickle(file_path=str(pfile))
    loaded = Preprocessor.from_pickle(file_path=str(pfile))
    assert isinstance(loaded, Preprocessor)
    assert loaded._fitted
    assert "node" in loaded.params

    post = Postprocessor(f=CenterReduceFunction())
    post._fitted = True
    post.params = {"node": jnp.array([[0.0, 1.0], [0.0, 2.0]])}
    qfile = tmp_path / "post.pkl"
    post.to_pickle(file_path=str(qfile))
    loaded2 = Postprocessor.from_pickle(file_path=str(qfile))
    assert isinstance(loaded2, Postprocessor)
    assert loaded2._fitted
    assert "node" in loaded2.params


# Identity (no-op) pre/post processing roundtrip
def test_identity_function_roundtrip_and_batch_consistency():
    # Fit preprocessors/postprocessors with IdentityFunction using the TestProblemLoader
    pre = Preprocessor(f=IdentityFunction())
    pre.fit_problem_loader(pb_loader, progress_bar=False)

    post = Postprocessor(f=IdentityFunction())
    post.fit_problem_loader(pb_loader, progress_bar=False)

    # iterate a few batches and test invertibility & batch-vs-single equivalence
    for pb_batch in pb_loader:
        # context roundtrip single <-> batch
        context_batch, _ = pb_batch.get_context()
        jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
        norm_batch, _ = pre.preprocess_batch(jax_context_batch)
        # convert back and forth to numpy graph
        norm_np = norm_batch.to_numpy_graph()
        norm_back = JaxGraph.from_numpy_graph(norm_np)
        denorm_batch, _ = pre.preprocess_inverse_batch(norm_back)
        denorm_np = denorm_batch.to_numpy_graph()

        # preprocessing is invertible for identity function
        np.testing.assert_allclose(denorm_np.feature_flat_array, context_batch.feature_flat_array, rtol=1e-6)

        # postprocess on zero decision -> gradient call should not crash
        norm_decision, _ = pb_batch.get_zero_decision()
        jax_norm_decision = JaxGraph.from_numpy_graph(norm_decision)
        # postprocess_batch shouldn't raise
        jax_decision, _ = post.postprocess_batch(jax_norm_decision)
        decision = jax_decision.to_numpy_graph()
        # ensure we can request gradient
        gradient, _ = pb_batch.get_gradient(decision=decision)
        assert gradient is not None


# CenterReduce: invertibility + numeric check on mean/std and gradient preconditioning
def test_center_reduce_preprocess_inverse_and_statistics():
    # Build "a" dataset used to compute params (2 samples, 2 features)
    a = jnp.array([[0.0, 0.0], [2.0, 2.0]])
    aux = [a]
    cr = CenterReduceFunction(epsilon=1e-8)
    params = cr.compute_params(None, aux)  # params[0]=mean, params[1]=std

    # Check that normalizing the SAME data 'a' yields mean ~0 and std ~1
    mask_a = jnp.array([1.0, 1.0])[:, None]  # shape (2,1) to avoid broadcasting issues
    normalized_a = cr.apply(params, a, mask_a)
    # Only keep valid rows (mask == 1)
    valid_norm_a = np.array(normalized_a).reshape(-1, normalized_a.shape[-1])
    means_a = np.nanmean(valid_norm_a, axis=0)
    stds_a = np.nanstd(valid_norm_a, axis=0)
    np.testing.assert_allclose(means_a, np.zeros(a.shape[1]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stds_a, np.ones(a.shape[1]), rtol=1e-6, atol=1e-6)

    # Prepare a larger array (contains the previous a plus another sample).
    array = jnp.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    mask = jnp.array([1.0, 1.0, 1.0])[:, None]  # shape (3,1)

    # Normalize and invert; apply_inverse must recover original values on valid rows
    normalized = cr.apply(params, array, mask)
    denorm = cr.apply_inverse(params, normalized, mask)
    np.testing.assert_allclose(np.array(denorm), np.array(array), rtol=1e-6, atol=1e-6)

    # Check gradient_inverse: should be equal to std + epsilon on valid rows
    grad_inv = cr.gradient_inverse(params, normalized, mask)
    std = np.array(params[1])  # per-feature std
    expected_grad_row = std + cr.epsilon
    # check each valid row
    grad_inv_valid = np.array(grad_inv).reshape(-1, grad_inv.shape[-1])
    for row in grad_inv_valid:
        np.testing.assert_allclose(row, expected_grad_row, rtol=1e-6, atol=1e-6)


def test_postprocessor_center_reduce_preconditioning_numeric():
    """
    Test precondition_gradient by constructing a small synthetic scenario
    where we know the std used by CenterReduceFunction, then check that
    preconditioning divides gradient by (std + eps).
    """
    # Build CenterReduceFunction params with std = [2.0, 0.5]
    cr = CenterReduceFunction(epsilon=1e-8)
    mean = jnp.array([10.0, -2.0], dtype=jnp.float32)
    std = jnp.array([2.0, 0.5], dtype=jnp.float32)

    # create two samples: mean - std and mean + std -> std will be as desired
    sample1 = (mean - std)
    sample2 = (mean + std)
    aux = [jnp.stack([sample1, sample2], axis=0)]  # shape (2, D)
    params_node = cr.compute_params(None, aux)  # returns stacked [mean, std] shape (2, D)

    # Assemble Postprocessor with these params manually
    post = Postprocessor(f=cr)
    post.params = {"node": params_node}
    post._fitted = True

    # Build minimal JaxGraphShape objects required by JaxGraph constructor
    # We'll set true_shape/current_shape.edges["node"] equal to the number of objects (2)
    n_objs = 2
    true_shape = JaxGraphShape(edges={"node": jnp.array(n_objs)}, addresses=jnp.array(0))
    current_shape = JaxGraphShape(edges={"node": jnp.array(n_objs)}, addresses=jnp.array(0))

    # --- Single-sample case ---
    feat = jnp.array([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)  # shape (N=2, D=2)
    non_fict = jnp.array([1.0, 1.0], dtype=jnp.float32)  # shape (N,)

    edge = JaxEdge(address_dict=None, feature_array=feat, feature_names=None, non_fictitious=non_fict)
    out_graph = JaxGraph(edges={"node": edge}, non_fictitious_addresses=non_fict, true_shape=true_shape, current_shape=current_shape)

    # gradient graph: suppose gradient is some known array
    grad_feat = jnp.array([[2.0, 4.0], [6.0, 8.0]], dtype=jnp.float32)
    grad_edge = JaxEdge(address_dict=None, feature_array=grad_feat, feature_names=None, non_fictitious=non_fict)
    grad_graph = JaxGraph(edges={"node": grad_edge}, non_fictitious_addresses=non_fict, true_shape=true_shape, current_shape=current_shape)

    # call precondition_gradient (single)
    prec_grad_graph, _ = post.precondition_gradient(out_graph, grad_graph, get_info=False)
    prec = np.array(prec_grad_graph.edges["node"].feature_array)

    # gradient should have been divided by (std + eps) per feature
    expected = np.array(grad_feat) / (np.array(std) + 1e-8)
    np.testing.assert_allclose(prec, expected, rtol=1e-6, atol=1e-6)

    # --- Batch version ---
    B = 3
    # create batched features: shape (B, N, D)
    out_feat_b = jnp.stack([feat] * B, axis=0)   # (B, N, D)
    grad_feat_b = jnp.stack([grad_feat] * B, axis=0)

    # non_fict must be batched: shape (B, N)
    non_fict_b = jnp.tile(non_fict[None, :], (B, 1))  # shape (B, N)

    # Batched edges: feature_array shape (B, N, D), non_fictitious shape must be (B, N)
    batched_edge_out = JaxEdge(address_dict=None, feature_array=out_feat_b, feature_names=None, non_fictitious=non_fict_b)
    batched_out_graph = JaxGraph(
        edges={"node": batched_edge_out},
        non_fictitious_addresses=non_fict_b,
        true_shape=true_shape,
        current_shape=current_shape,
    )

    batched_edge_grad = JaxEdge(address_dict=None, feature_array=grad_feat_b, feature_names=None, non_fictitious=non_fict_b)
    batched_grad_graph = JaxGraph(
        edges={"node": batched_edge_grad},
        non_fictitious_addresses=non_fict_b,
        true_shape=true_shape,
        current_shape=current_shape,
    )

    prec_b_graph, _ = post.precondition_gradient_batch(batched_out_graph, batched_grad_graph, get_info=False)
    prec_b = np.array(prec_b_graph.edges["node"].feature_array)

    # expected broadcasting divides across last dim (D)
    expected_b = np.array(grad_feat_b) / (np.array(std) + 1e-8)
    np.testing.assert_allclose(prec_b, expected_b, rtol=1e-6, atol=1e-6)


# CDFPWLinearFunction basic invertibility test and batch handling
def test_cdfpw_linear_invertibility_small():
    device = jax.devices("cpu")[0]

    pre = Preprocessor(f=CDFPWLinearFunction(n_breakpoints=10))
    pre.fit_problem_loader(pb_loader, progress_bar=False, device=device)

    post = Postprocessor(f=CDFPWLinearFunction(n_breakpoints=10))
    post.fit_problem_loader(pb_loader, progress_bar=False, device=device)

    pb_batch = next(iter(pb_loader))
    context_batch, _ = pb_batch.get_context()
    jax_context = JaxGraph.from_numpy_graph(context_batch, device=device)

    norm_batch, _ = pre.preprocess_batch(jax_context)
    # inverse should restore original (approximately)
    recovered_batch, _ = pre.preprocess_inverse_batch(norm_batch)
    np.testing.assert_allclose(np.array(recovered_batch.to_numpy_graph().feature_flat_array), np.array(context_batch.feature_flat_array), rtol=1e-5)


# Cleanup test: ensure temp dirs created during tests are removed
def test_save_and_load_to_temp_folder():
    path = "tmp_path/energnn/normalizer"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

    pre = Preprocessor(f=CenterReduceFunction())
    pre.fit_problem_loader(pb_loader, progress_bar=False)
    assert pre._fitted

    post = Postprocessor(f=CenterReduceFunction())
    post.fit_problem_loader(pb_loader, progress_bar=False)
    assert post._fitted

    pre_file = os.path.join(path, "preprocessor.pkl")
    post_file = os.path.join(path, "postprocessor.pkl")
    pre.to_pickle(file_path=str(pre_file))
    post.to_pickle(file_path=str(post_file))

    loaded_pre = Preprocessor.from_pickle(file_path=str(pre_file))
    loaded_post = Postprocessor.from_pickle(file_path=str(post_file))

    assert isinstance(loaded_pre, Preprocessor)
    assert isinstance(loaded_post, Postprocessor)
    assert loaded_pre._fitted
    assert loaded_post._fitted
    shutil.rmtree("tmp_path", ignore_errors=True)
