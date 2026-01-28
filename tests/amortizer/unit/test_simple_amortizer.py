#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from energnn.amortizer.simple_amortizer import SimpleAmortizer
from energnn.graph.jax import JaxGraph
from energnn.normalizer import Preprocessor, Postprocessor

from tests.utils import TestProblemLoader
from tests.amortizer.unit.utils import (
    Decision,
    FakeJaxGraph,
    FakeGNN,
    FakeTracker,
    FakePostprocessor,
    FakePreprocessor,
    FakeProblemLoader,
    FakeProblemBatch,
    FakeRegistry
)


@pytest.fixture(autouse=True)
def patch_graph_helpers(monkeypatch):
    """
    Replace separate_graphs and JaxGraph.from_numpy_graph with simplified versions
    for unit tests so we can use plain dict/list as "numpy graphs".
    """
    monkeypatch.setattr("energnn.graph.separate_graphs", lambda g: [g])
    # from_numpy_graph should wrap any numpy "graph" into our FakeJaxGraph
    monkeypatch.setattr(
        "energnn.graph.jax.JaxGraph.from_numpy_graph",
        lambda g, device=None: FakeJaxGraph(numpy_graph=g, feature_flat_array=getattr(g, "feature_flat_array", [[0.0]])),
    )
    yield


# create a tiny loader
tiny_loader = TestProblemLoader(
    dataset_size=1,
    n_batch=1,
    context_edge_params={
        "node": {"n_obj": 1, "feature_list": ["a"], "address_list": ["0"]},
    },
    oracle_edge_params={
        "node": {"n_obj": 1, "feature_list": ["e"]},
    },
    n_addr=1,
    shuffle=False,
)
pb_batch = next(iter(tiny_loader))
loader = FakeProblemLoader([pb_batch])


def build_amortizer_with_fakes(**kwargs):
    """
    Helper to build a SimpleAmortizer with fake components and a simple optimizer.
    Returns amortizer, fakes dict.
    """
    gnn = kwargs.get("../../gnn", FakeGNN())
    pre = kwargs.get("pre", FakePreprocessor())
    post = kwargs.get("post", FakePostprocessor())
    optimizer = kwargs.get("optimizer", optax.sgd(learning_rate=0.01))
    amort = SimpleAmortizer(gnn=gnn, preprocessor=pre, postprocessor=post, optimizer=optimizer, progress_bar=False,
                            project_name="test_project", run_id="test_run")
    return amort, {"gnn": gnn, "pre": pre, "post": post, "optimizer": optimizer}


def test_init_calls_fit_and_sets_state():
    amort, fakes = build_amortizer_with_fakes()

    rngs = jax.random.PRNGKey(0)
    # call init
    amort.init(rngs=rngs, loader=loader, problem_cfg=None)

    # assert preprocessors were fit
    fakes["pre"].fit_problem_loader.assert_called()
    fakes["post"].fit_problem_loader.assert_called()

    # params/op_state initialization
    assert isinstance(amort.params, dict)
    assert amort.opt_state is not None
    assert amort.initialized is True
    assert amort.train_step == 0


def test_train_raises_if_not_initialized():
    amort, _ = build_amortizer_with_fakes()
    # Ensure not initialized
    amort.initialized = False
    with pytest.raises(RuntimeError):
        amort.train(
            train_loader=None,
            val_loader=None,
            problem_cfg=None,
            n_epochs=1,
            registry=FakeRegistry(),
            tracker=FakeTracker(),
        )


def test_forward_delegates_to_pre_and_post_and_gnn():
    gnn = FakeGNN(apply_return=(FakeJaxGraph(feature_flat_array=[[5.0, 6.0]]), {"g": 1}))
    pre = FakePreprocessor(preprocess_return=(FakeJaxGraph(feature_flat_array=[[1.0, 2.0]]), {"p": 1}))
    post = FakePostprocessor(postprocess_return=(FakeJaxGraph(feature_flat_array=[[7.0, 8.0]]), {"pp": 1}))
    amort = SimpleAmortizer(gnn=gnn, preprocessor=pre, postprocessor=post, optimizer=optax.sgd(0.01), progress_bar=False,
                            project_name="test_project", run_id="test_run")

    # Create a fake context (any object)
    context = SimpleNamespace(feature_flat_array=[[0.0]])
    jax_context = JaxGraph.from_numpy_graph(context)
    # call forward
    result_graph, infos = amort.forward(params={"p": {}}, context=jax_context, get_info=True)

    # pre, gnn.apply, post should be called
    assert pre.preprocess.called
    assert gnn.apply_called
    assert post.postprocess.called

    # outputs are from postprocessor fake
    assert isinstance(result_graph, FakeJaxGraph)
    assert isinstance(infos, dict)
    assert "preprocess" in infos and "gnn" in infos and "postprocess" in infos


def test_infer_and_infer_batch_delegate_to_forward_without_jit():
    """
    Ensure infer (single) and infer_batch (batched) delegate to forward.
    Use a simple batched pytree for context so vmap sees a consistent leading axis.
    We call the underlying __wrapped__ (non-jitted) functions because the methods are jit-wrapped.
    """

    # Build a simple batched pytree where every leaf has leading batch axis.
    B = 4
    # Example leaves: two features per object, batch size B
    batch_ctx = {
        "feat": jax.numpy.zeros((B, 2), dtype=jax.numpy.float32),
        "mask": jax.numpy.ones((B,), dtype=jax.numpy.float32),
    }
    # Single context (no batch axis) is the first element of batch_ctx
    single_ctx = jax.tree_map(lambda x: x[0], batch_ctx)

    # Build amortizer and monkeypatch forward with a simple Python function
    amort, _ = build_amortizer_with_fakes()

    # fake_forward will be called either directly (infer) or via vmap (infer_batch).
    # It must accept (params, context, get_info) and return (decision, info).
    def fake_forward(params, context, get_info):
        return {"dummy_decision": 0}, {"fwd": 1}

    amort.forward = fake_forward
    amort.params = {"encoder": {}, "coupler": {}, "decoder": {}}

    # Call non-jitted infer (single)
    infer_res = amort.infer.__wrapped__(amort, single_ctx, True)
    assert isinstance(infer_res, tuple)
    # info from fake_forward for single should have 'fwd' == 1
    assert infer_res[1]["fwd"] == 1

    # Call non-jitted infer_batch (batched) — internally calls vmap(fake_forward, in_axes=(None,0,None))
    infer_batch_res = amort.infer_batch.__wrapped__(amort, batch_ctx, True)
    assert isinstance(infer_batch_res, tuple)
    infos = infer_batch_res[1]
    assert "fwd" in infos
    fwd_val = infos["fwd"]
    # If vmap collected scalar infos into an array, check leading dim equals B

    if hasattr(fwd_val, "shape"):
        assert int(np.asarray(fwd_val).shape[0]) == B
    else:
        # fallback: ensure scalar present
        assert fwd_val == 1


def test_run_evaluation_saves_on_improvement(monkeypatch, tmp_path):
    amort, _ = build_amortizer_with_fakes()
    registry = FakeRegistry()
    tracker = FakeTracker()

    # monkeypatch eval to return a metric better than best_metrics
    amort.best_metrics = 1.0
    amort.save = MagicMock()
    amort.eval = MagicMock(return_value=(0.5, {"dummy": np.array([0.0])}))

    amort.run_evaluation(val_loader=None, cfg=None, tracker=tracker, registry=registry)

    # since metric improved, register_trainer should have been called
    registry.register_trainer.assert_called_with(trainer=amort, best=True)
    assert amort.best_metrics == 0.5


def test_save_and_load_roundtrip(tmp_path):
    amort, _ = build_amortizer_with_fakes()
    # set an attribute to check roundtrip
    amort.progress_bar = False
    amort.save(name="tmp_amort", directory=str(tmp_path))
    # saved file name should exist
    file_path = os.path.join(str(tmp_path), "tmp_amort")
    assert os.path.exists(file_path)

    # load back
    loaded = SimpleAmortizer.load(file_path)
    assert isinstance(loaded, SimpleAmortizer)
    # loaded should have similar attribute
    assert hasattr(loaded, "progress_bar")


def test_training_step_returns_flat_numpy_infos(monkeypatch):
    # This test ensures training_step orchestrates calls and flattens infos to numpy arrays
    # Build amortizer and monkeypatch internals
    gnn = FakeGNN(apply_return=(FakeJaxGraph(feature_flat_array=[[2.0, 2.0]]), {"g": 1}))
    pre = FakePreprocessor(preprocess_return=(FakeJaxGraph(feature_flat_array=[[1.0, 1.0]]), {"p": 1}))
    post = FakePostprocessor(postprocess_return=(FakeJaxGraph(feature_flat_array=[[0.0, 0.0]]), {"pp": 1}))
    amort = SimpleAmortizer(gnn=gnn, preprocessor=pre, postprocessor=post, optimizer=optax.sgd(0.01), progress_bar=False,
                            project_name="test_project", run_id="test_run")

    # create a fake problem batch with simple numpy-like context and gradient
    context = SimpleNamespace(feature_flat_array=[[0.0]])
    # For get_gradient, return a numpy-graph like object (we will wrap via JaxGraph.from_numpy_graph)
    grad = SimpleNamespace(feature_flat_array=[[0.5, 0.5]])
    pb = FakeProblemBatch(context=context, decision_structure={"node": {"a": 1}}, gradient=grad)

    # patch forward_batch to return a FakeJaxGraph and info
    amort.forward_batch = MagicMock(return_value=(FakeJaxGraph(feature_flat_array=[[3.0, 4.0]]), {"fwd_b": np.array([1.0])}))
    # patch update_params to return same params,opt_state and some infos
    amort.update_params = MagicMock(return_value=({"p": {}}, {"opt": 1}, {"loss": 0.1}))

    # monkeypatch JaxGraph.from_numpy_graph to ensure gradient/context wrapping done by fixture patch_graph_helpers
    # call training_step
    amort.params = {"p": {}}
    amort.opt_state = {"opt": 1}
    infos = amort.training_step(problem_batch=pb, cfg=None, get_info=True)

    # infos should be a dict with numpy arrays (flat keys)
    assert isinstance(infos, dict)
    for k, v in infos.items():
        assert isinstance(v, np.ndarray)


def test_eval_aggregates_metrics_and_infos(monkeypatch):
    amort, _ = build_amortizer_with_fakes()
    # define two batches whose eval_step returns metrics and infos
    pb1 = FakeProblemBatch(context=SimpleNamespace(), decision_structure={"node": {"a": 1}}, metrics=[1.0])
    pb2 = FakeProblemBatch(context=SimpleNamespace(), decision_structure={"node": {"a": 1}}, metrics=[3.0])
    loader = FakeProblemLoader([pb1, pb2])

    # stub eval_step to return arrays and info arrays
    amort.eval_step = MagicMock(
        side_effect=[
            (np.array([1.0]), {"a": np.array([10.0])}),
            (np.array([3.0]), {"a": np.array([30.0])}),
        ]
    )

    metrics, infos = amort.eval(loader, cfg=None)
    # metrics = mean([1.,3.]) = 2.0
    assert pytest.approx(metrics) == 2.0
    assert "a" in infos
    assert infos["metrics"] == metrics


def test_update_params_numeric_gradient_application():
    """
    Numeric test of SimpleAmortizer.update_params:
    - Make _apply_model return Decision(feature_flat_array = context * params['w'])
    - Use preprocessor.preprocess_batch to return a batch of contexts:
          contexts = [[1,1], [2,2]]  -> flattened mean = 1.5
    - Starting params['w'] = 2.0, optimizer = sgd(0.1)
    - Expected grad w.r.t w = mean(context_elements) = 1.5
    - Expected new_w = 2.0 - 0.1 * 1.5 = 1.85
    """

    # Build a minimal SimpleAmortizer with dummy components
    # Instances of Preprocessor/Postprocessor that expose preprocess_batch/precondition_gradient_batch needed
    pre = Preprocessor.__new__(Preprocessor)
    post = Postprocessor.__new__(Postprocessor)

    amort = SimpleAmortizer(gnn=None, preprocessor=pre, postprocessor=post, optimizer=optax.sgd(0.1), progress_bar=False,
                            project_name="test_project", run_id="test_run")

    # Monkeypatch preprocessor.preprocess_batch to return our batched contexts (B=2, D=2).
    # This function must return (norm_context, info). norm_context will be iterated by vmap axis=0.
    contexts = jnp.array([[1.0, 1.0], [2.0, 2.0]])  # flattened mean = (1+1+2+2)/4 = 1.5

    amort.preprocessor.preprocess_batch = lambda ctx, get_info=False: (contexts, {})

    # Provide a fake postprocessor.precondition_gradient_batch which returns ones of same shape
    # Accepts (norm_decision, gradient) and returns (prec_grad, {})
    def fake_prec(norm_decision, gradient, get_info=False):
        # norm_decision.feature_flat_array shape will be (B, 1, D) in our construction below
        ones = jnp.ones_like(norm_decision.feature_flat_array)
        return Decision(ones), {}

    amort.postprocessor.precondition_gradient_batch = fake_prec

    # Monkeypatch SimpleAmortizer._apply_model to produce Decision(feature_flat_array = context[None,:] * params['w'])
    def fake_apply_model(params, context_elem, get_info):
        # context_elem: one sample context, shape (D,)
        # params: dict with key 'w' (scalar)
        w = params["w"]
        # decision feature array per sample: shape (1, D)
        feat = (context_elem * w)[None, :]  # add object-dimension (e.g., N=1)
        return Decision(feat), {}

    amort._apply_model = fake_apply_model

    # Prepare params, opt_state, context and gradient arguments for update_params
    params = {"w": jnp.array(2.0)}  # initial param
    optimizer = amort.optimizer
    opt_state = optimizer.init(params)

    # context argument passed to update_params is not used by our patched preprocess_batch, so can be any object
    dummy_context = JaxGraph.__new__(JaxGraph)

    # gradient argument is ignored by fake_prec, provide a dummy JaxGraph-like object:
    dummy_grad = JaxGraph.__new__(JaxGraph)

    # Call the non-jitted version to avoid complications with JIT tracing in unit test
    new_params, new_opt_state, infos = amort.update_params.__wrapped__(
        amort, params, opt_state, dummy_context, dummy_grad, True
    )

    # Expected calculations:
    flat_mean = float(jnp.mean(contexts))  # 1.5
    expected_grad = flat_mean
    expected_new_w = float(params["w"]) - 0.1 * expected_grad  # sgd lr=0.1

    # Validate new parameter close to expected
    assert np.allclose(float(new_params["w"]), expected_new_w, rtol=1e-6, atol=1e-6)

    # Infos contains 'loss' and grads / updates metrics (pytrees) - ensure they exist and are numeric
    assert "loss" in infos
    assert "grads/l2_norm" in infos
    # grads/l2_norm should be positive
    assert float(infos["grads/l2_norm"]) >= 0.0
