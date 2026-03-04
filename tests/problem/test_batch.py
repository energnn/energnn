import pytest
import jax
import jax.numpy as jnp
import numpy as np

from energnn.problem.batch import ProblemBatch
from energnn.graph import GraphStructure, EdgeStructure
from energnn.graph.jax import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader


@pytest.fixture(scope="module")
def pb_loader():
    # Small deterministic loader used in other tests of the repo
    return TestProblemLoader(seed=0)


@pytest.fixture(scope="module")
def pb_batch(pb_loader):
    # grab one batch instance from the loader
    return next(iter(pb_loader))


class StubProblemBatch(ProblemBatch):
    """Minimal concrete ProblemBatch for testing the base interface."""

    def __init__(self, context=None, decision=None):
        self.context = context
        self.decision = decision

    @property
    def context_structure(self) -> GraphStructure:
        return GraphStructure(edges={})

    @property
    def decision_structure(self) -> GraphStructure:
        return GraphStructure(edges={})

    def get_context(self, get_info: bool = False):
        info = {"cinfo": True} if get_info else {}
        return self.context, info

    def get_gradient(self, *, decision, get_info: bool = False):
        info = {"ginfo": "ok"} if get_info else {}
        return decision, info

    def get_metrics(self, *, decision, get_info: bool = False):
        info = {"minfo": "m"} if get_info else {}
        return [0.0], info

    def get_zero_decision(self, get_info: bool = False):
        info = {"zinfo": 0} if get_info else {}
        return self.decision, info

    def get_decision_structure(self) -> dict:
        """Utility method commonly expected in subclasses."""
        zero_decision, _ = self.get_zero_decision(get_info=False)
        structure = {}
        for edge_key, edge in zero_decision.edges.items():
            if edge.feature_names is not None:
                structure[edge_key] = {name: int(idx) for name, idx in edge.feature_names.items()}
        return structure


def test_problembatch_is_abstract():
    """ProblemBatch is an abstract base class and cannot be instantiated."""
    with pytest.raises(TypeError):
        ProblemBatch()


def test_methods_return_tuple_and_info():
    """Check each required method returns (Data, dict) and handles get_info flag."""
    dummy_graph = jax.tree.map(lambda x: x, {"edges": {}})
    pb = StubProblemBatch(context=dummy_graph, decision=dummy_graph)

    # get_context
    _, info = pb.get_context(get_info=False)
    assert info == {}
    _, info = pb.get_context(get_info=True)
    assert info == {"cinfo": True}

    # get_gradient
    _, info = pb.get_gradient(decision=dummy_graph, get_info=False)
    assert info == {}
    _, info = pb.get_gradient(decision=dummy_graph, get_info=True)
    assert info == {"ginfo": "ok"}

    # get_metrics
    metrics, info = pb.get_metrics(decision=dummy_graph, get_info=True)
    assert isinstance(metrics, list)
    assert info == {"minfo": "m"}


@pytest.mark.parametrize(
    "feature_names, expected_values",
    [
        ({"a": 0, "b": 1}, {"a": 0, "b": 1}),
        ({"a": jnp.array(0), "b": np.int64(2)}, {"a": 0, "b": 2}),
    ],
)
def test_get_decision_structure_conversions(feature_names, expected_values):
    """get_decision_structure should correctly convert various int-like types to native ints."""
    edge = JaxEdge(
        address_dict=None,
        feature_array=jnp.zeros((1, 2)),
        feature_names=feature_names,
        non_fictitious=jnp.ones((1,)),
    )
    decision = JaxGraph(edges={"node": edge}, non_fictitious_addresses=jnp.array([]), true_shape=None, current_shape=None)
    pb = StubProblemBatch(decision=decision)

    ds = pb.get_decision_structure()
    assert ds["node"] == expected_values
    for val in ds["node"].values():
        assert isinstance(val, int)


def test_get_gradient_shapes_match_decision(pb_batch):
    """get_gradient must return a gradient Graph with the same edge keys and shapes as the decision input."""
    # pb_batch fixture is already a valid TestProblemBatch (concrete ProblemBatch)
    ctx, _ = pb_batch.get_context()

    # Use oracle as a valid decision to test gradient computation
    oracle, _ = pb_batch.get_oracle()
    grad, _ = pb_batch.get_gradient(decision=oracle, get_info=False)

    # Check edge keys and shapes
    assert set(oracle.edges.keys()) == set(grad.edges.keys())

    for key in oracle.edges:
        dec_arr = oracle.edges[key].feature_array
        grad_arr = grad.edges[key].feature_array
        if dec_arr is not None:
            assert dec_arr.shape == grad_arr.shape
        else:
            assert grad_arr is None
