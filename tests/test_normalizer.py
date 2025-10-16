#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import os
import shutil

import jax
import numpy as np

from energnn.graph.jax import JaxGraph
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


def test_identity_function():

    preprocessor = Preprocessor(f=IdentityFunction())
    preprocessor.fit_problem_loader(pb_loader)

    postprocessor = Postprocessor(f=IdentityFunction())
    postprocessor.fit_problem_loader(pb_loader)

    for pb_batch in pb_loader:
        context, _ = pb_batch.get_context()
        jax_context = JaxGraph.from_numpy_graph(context)
        norm_jax_context, _ = preprocessor.preprocess_batch(jax_context)
        norm_context = norm_jax_context.to_numpy_graph()

        norm_jax_context = JaxGraph.from_numpy_graph(norm_context)
        denorm_jax_context, _ = preprocessor.preprocess_inverse_batch(norm_jax_context)
        denorm_context = denorm_jax_context.to_numpy_graph()

        # Assert preprocessing is invertible.
        np.testing.assert_almost_equal(denorm_context.feature_flat_array, context.feature_flat_array)

        norm_decision, _ = pb_batch.get_zero_decision()
        jax_norm_decision = JaxGraph.from_numpy_graph(norm_decision)
        jax_decision, _ = postprocessor.postprocess_batch(jax_norm_decision)
        decision = jax_decision.to_numpy_graph()

        gradient, _ = pb_batch.get_gradient(decision=decision)


def test_center_reduce_function():

    preprocessor = Preprocessor(f=CenterReduceFunction())
    preprocessor.fit_problem_loader(pb_loader)

    postprocessor = Postprocessor(f=CenterReduceFunction())
    postprocessor.fit_problem_loader(pb_loader)

    for pb_batch in pb_loader:
        context, _ = pb_batch.get_context()
        jax_context = JaxGraph.from_numpy_graph(context)
        norm_jax_context, _ = preprocessor.preprocess_batch(jax_context)
        norm_context = norm_jax_context.to_numpy_graph()

        norm_jax_context = JaxGraph.from_numpy_graph(norm_context)
        denorm_jax_context, _ = preprocessor.preprocess_inverse_batch(norm_jax_context)
        denorm_context = denorm_jax_context.to_numpy_graph()

        # Assert preprocessing is invertible.
        np.testing.assert_almost_equal(denorm_context.feature_flat_array, context.feature_flat_array)

        norm_decision, _ = pb_batch.get_zero_decision()
        jax_norm_decision = JaxGraph.from_numpy_graph(norm_decision)
        jax_decision, _ = postprocessor.postprocess_batch(jax_norm_decision)
        decision = jax_decision.to_numpy_graph()

        gradient, _ = pb_batch.get_gradient(decision=decision)
        jax_gradient = JaxGraph.from_numpy_graph(gradient)
        prec_jax_gradient, _ = postprocessor.precondition_gradient_batch(jax_norm_decision, jax_gradient)
        prec_gradient = prec_jax_gradient.to_numpy_graph()

        norm_decision.feature_flat_array -= prec_gradient.feature_flat_array
        jax_norm_decision = JaxGraph.from_numpy_graph(norm_decision)
        jax_decision, _ = postprocessor.postprocess_batch(jax_norm_decision)
        decision = jax_decision.to_numpy_graph()

        metrics, _ = pb_batch.get_metrics(decision=decision)
        np.testing.assert_almost_equal(metrics, np.zeros(4))


def test_cdf_function():

    device = jax.devices("cpu")[0]

    preprocessor = Preprocessor(f=CDFPWLinearFunction())
    preprocessor.fit_problem_loader(pb_loader, device=device)

    postprocessor = Postprocessor(f=CDFPWLinearFunction())
    postprocessor.fit_problem_loader(pb_loader, device=device)

    for pb_batch in pb_loader:
        context, _ = pb_batch.get_context()
        jax_context = JaxGraph.from_numpy_graph(context, device=device)
        norm_jax_context, _ = preprocessor.preprocess_batch(jax_context)
        norm_context = norm_jax_context.to_numpy_graph()

        norm_jax_context = JaxGraph.from_numpy_graph(norm_context, device=device)
        denorm_jax_context, _ = preprocessor.preprocess_inverse_batch(norm_jax_context)
        denorm_context = denorm_jax_context.to_numpy_graph()

        # Assert preprocessing is invertible.
        np.testing.assert_almost_equal(denorm_context.feature_flat_array, context.feature_flat_array)

        norm_decision, _ = pb_batch.get_zero_decision()
        jax_norm_decision = JaxGraph.from_numpy_graph(norm_decision, device=device)
        jax_decision, _ = postprocessor.postprocess_batch(jax_norm_decision)
        decision = jax_decision.to_numpy_graph()

        gradient, _ = pb_batch.get_gradient(decision=decision)


def test_load_file_path():

    path = "tmp/argo/normalizer"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

    preprocessor = Preprocessor(f=CenterReduceFunction())
    assert not preprocessor._fitted
    preprocessor.fit_problem_loader(pb_loader)
    assert preprocessor._fitted
    postprocessor = Postprocessor(f=CenterReduceFunction())
    assert not postprocessor._fitted
    postprocessor.fit_problem_loader(pb_loader)
    assert postprocessor._fitted

    # Saving normalizers
    preprocessor_file_path = os.path.join(path, "preprocessor.pkl")
    postprocessor_file_path = os.path.join(path, "postprocessor.pkl")
    preprocessor.to_pickle(file_path=preprocessor_file_path)
    postprocessor.to_pickle(file_path=postprocessor_file_path)

    # Loading saved normalizers
    saved_preprocessor = Preprocessor.from_pickle(file_path=preprocessor_file_path)
    saved_postprocessor = Postprocessor.from_pickle(file_path=postprocessor_file_path)

    assert isinstance(saved_preprocessor, Preprocessor)
    assert isinstance(saved_postprocessor, Postprocessor)
    assert saved_preprocessor._fitted
    assert saved_postprocessor._fitted
    shutil.rmtree("tmp", ignore_errors=True)
