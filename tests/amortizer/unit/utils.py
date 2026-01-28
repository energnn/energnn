#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock


class FakeJaxGraph:
    """
    Minimal jax-graph-like object used in tests.
    Exposes feature_flat_array and to_numpy_graph().
    """

    def __init__(self, numpy_graph=None, feature_flat_array=None):
        # numpy_graph can be any object representing the original numpy Graph
        self._numpy_graph = numpy_graph if numpy_graph is not None else {}
        # default small feature_flat_array if none provided
        if feature_flat_array is None:
            self.feature_flat_array = jnp.array([[0.0, 0.0]])
        else:
            self.feature_flat_array = jnp.asarray(feature_flat_array)

    def to_numpy_graph(self):
        return self._numpy_graph


class FakeProblemBatch:
    """
    Minimal ProblemBatch double.
    """

    def __init__(self, context, decision_structure, gradient=None, metrics=None):
        self._context = context
        self._decision_structure = decision_structure
        # gradient / metrics returned by get_gradient/get_metrics
        self._gradient = gradient if gradient is not None else {"dummy": 0}
        self._metrics = metrics if metrics is not None else [0.0]

    def get_context(self, get_info=False):
        # return a simple "numpy" context object and empty info
        return self._context, {}

    def get_decision_structure(self):
        return self._decision_structure

    def get_gradient(self, *, decision, cfg=None, get_info=False):
        # return a simple numpy-graph-like object and empty info
        return self._gradient, {}

    def get_metrics(self, *, decision, get_info=False, cfg=None):
        return np.array(self._metrics), {}


class FakeProblemLoader:
    """
    Iterable that yields FakeProblemBatch instances.
    """

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __next__(self):
        return self._batches.pop(0)

    def __len__(self):
        return len(self._batches)


class FakeGNN:
    """
    Fake GNN with init and apply minimal implementations.
    """

    def __init__(self, apply_return=None, init_return=None):
        # apply_return: what apply should return (JaxGraph, info)
        self.apply_return = apply_return if apply_return is not None else (FakeJaxGraph(feature_flat_array=[[1.0, 2.0]]), {})
        self.init_return = init_return if init_return is not None else {"encoder": {}, "coupler": {}, "decoder": {}}
        self.init_called = False
        self.apply_called = False

    def init(self, rngs, context, out_structure):
        self.init_called = True
        return self.init_return

    def apply(self, params, context, get_info=False):
        self.apply_called = True
        return self.apply_return


class FakePreprocessor:
    def __init__(self, preprocess_return=None, preprocess_batch_return=None):
        self.fit_problem_loader = MagicMock()
        self.preprocess_return = preprocess_return if preprocess_return is not None else (FakeJaxGraph(), {})
        self.preprocess_batch_return = preprocess_batch_return if preprocess_batch_return is not None else (FakeJaxGraph(), {})
        self.preprocess = MagicMock(side_effect=lambda context, get_info=False: self.preprocess_return)
        self.preprocess_batch = MagicMock(side_effect=lambda context, get_info=False: self.preprocess_batch_return)


class FakePostprocessor:
    def __init__(self, postprocess_return=None, postprocess_batch_return=None, prec_return=None, prec_batch_return=None):
        self.fit_problem_loader = MagicMock()
        self.postprocess_return = postprocess_return if postprocess_return is not None else (FakeJaxGraph(), {})
        self.postprocess_batch_return = (
            postprocess_batch_return if postprocess_batch_return is not None else (FakeJaxGraph(), {})
        )
        self.postprocess = MagicMock(side_effect=lambda decision, get_info=False: self.postprocess_return)
        self.postprocess_batch = MagicMock(side_effect=lambda decision, get_info=False: self.postprocess_batch_return)
        # preconditioning returns gradient-like JaxGraph
        self.prec_return = prec_return if prec_return is not None else (FakeJaxGraph(feature_flat_array=[[1.0, 1.0]]), {})
        self.prec_batch_return = (
            prec_batch_return if prec_batch_return is not None else (FakeJaxGraph(feature_flat_array=[[1.0, 1.0]]), {})
        )
        self.precondition_gradient = MagicMock(side_effect=lambda out_graph, grad_graph, get_info=False: self.prec_return)
        self.precondition_gradient_batch = MagicMock(
            side_effect=lambda out_graph, grad_graph, get_info=False: self.prec_batch_return
        )


class FakeStorage:
    def __init__(self):
        self.upload = MagicMock()
        self.download = MagicMock()
        self.delete = MagicMock()


class FakeTracker:
    def __init__(self):
        self.run_append = MagicMock()


# register a simple Decision class as a pytree so jax.vmap can stack/unstack it.
class Decision:
    def __init__(self, feature_flat_array):
        self.feature_flat_array = feature_flat_array


def _dec_flatten(decision):
    # children must be a tuple/list of arrays
    return (decision.feature_flat_array,), None


def _dec_unflatten(aux_data, children):
    return Decision(children[0])


jax.tree_util.register_pytree_node(Decision, _dec_flatten, _dec_unflatten)
