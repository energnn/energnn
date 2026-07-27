# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from energnn.problem.example import LinearSystemContextConverter, LinearSystemOracleConverter
from energnn.problem.example.linear_system import (
    LINEAR_SYSTEM_CONTEXT_STRUCTURE,
    LINEAR_SYSTEM_DECISION_STRUCTURE,
    _generate_sparse_linear_system,
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
