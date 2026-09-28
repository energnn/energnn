# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph

from energnn.graph.backend import NumpyBackend
from energnn.problem.example import LinearSystemContextConverter, LinearSystemOracleConverter
from energnn.problem.example.linear_system import (
    LINEAR_SYSTEM_CONTEXT_STRUCTURE,
    LINEAR_SYSTEM_DECISION_STRUCTURE,
    LinearSystemProblemGenerator,
    _draw_n_lines,
    _generate_sparse_linear_system,
    _max_lines,
)


def test_context_converter_matches_system():
    np.random.seed(0)
    B, P, theta = _generate_sparse_linear_system(8, 12)
    n = P.shape[0]

    graph = LinearSystemContextConverter()(B=B, P=P)

    # Bus addresses are the integers 0..n-1, in table order.
    bus = graph.hyper_edge_sets["bus"]
    assert bus.port_dict["id"].astype(int).tolist() == list(range(n))
    assert np.allclose(bus.feature_array.ravel(), P)
    assert int(graph.true_shape.addresses) == n

    # Line ports point to valid bus addresses and carry the off-diagonal susceptances.
    line = graph.hyper_edge_sets["line"]
    rows, cols = np.nonzero(np.triu(B, k=1))
    assert line.port_dict["from"].astype(int).tolist() == rows.tolist()
    assert line.port_dict["to"].astype(int).tolist() == cols.tolist()
    assert np.allclose(line.feature_array.ravel(), -B[rows, cols])


def test_oracle_converter_carries_only_features():
    np.random.seed(0)
    _, _, theta = _generate_sparse_linear_system(8, 12)

    graph = LinearSystemOracleConverter()(theta=theta)

    bus = graph.hyper_edge_sets["bus"]
    assert bus.port_dict is None
    assert np.allclose(bus.feature_array.ravel(), theta)
    # Oracles carry no ports, hence an empty address registry.
    assert int(graph.true_shape.addresses) == 0


def test_structures_derived_from_converters():
    assert LINEAR_SYSTEM_CONTEXT_STRUCTURE == LinearSystemContextConverter().get_structure()
    assert LINEAR_SYSTEM_DECISION_STRUCTURE == LinearSystemOracleConverter().get_structure()

    context_sets = LINEAR_SYSTEM_CONTEXT_STRUCTURE.hyper_edge_sets
    assert context_sets["line"].port_list == ["from", "to"]
    assert context_sets["bus"].feature_list == ["active_power_injection"]
    assert LINEAR_SYSTEM_DECISION_STRUCTURE.hyper_edge_sets["bus"].port_list is None


# ---------------------------------------------------------------------------
# Sparsity and connectivity of generated systems
# ---------------------------------------------------------------------------


def _is_connected(B: np.ndarray) -> bool:
    adjacency = (B != 0) & ~np.eye(B.shape[0], dtype=bool)
    n_components, _ = scipy.sparse.csgraph.connected_components(scipy.sparse.csr_matrix(adjacency), directed=False)
    return n_components == 1


@pytest.mark.parametrize("n", [2, 3, 5, 16, 64])
def test_generated_system_is_connected(n):
    np.random.seed(n)
    for _ in range(50):
        m = _draw_n_lines(n)
        B, _, _ = _generate_sparse_linear_system(n, m)
        assert n - 1 <= m <= _max_lines(n)
        assert np.count_nonzero(np.triu(B, k=1)) == m
        assert _is_connected(B)


@pytest.mark.parametrize("n_max", [3, 4])
def test_max_lines_never_exceeds_complete_graph(n_max):
    assert _max_lines(n_max) <= n_max * (n_max - 1) // 2
    assert _max_lines(2) == 1


def test_generated_systems_have_realistic_mean_degree():
    generator = LinearSystemProblemGenerator(seed=0, n_max=64)
    degrees = []
    for _ in range(300):
        problem = generator.generate_problem(backend=NumpyBackend())
        n = problem.context.hyper_edge_sets["bus"].n_obj
        m = problem.context.hyper_edge_sets["line"].n_obj
        assert m <= _max_lines(n)
        degrees.append(2 * m / n)
    assert 2.0 < np.mean(degrees) < 4.0


def test_problem_batch_pads_lines_to_max_lines():
    n_max = 16
    generator = LinearSystemProblemGenerator(seed=0, n_max=n_max)
    batch = generator.generate_problem_batch(batch_size=4)
    line = batch.context.hyper_edge_sets["line"]
    assert line.n_obj == _max_lines(n_max)
    assert batch.context.hyper_edge_sets["bus"].n_obj == n_max
