# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph import Graph, HyperEdgeSet, JaxBackend, NumpyBackend
from energnn.trainer import get_graph_statistics, get_series_statistics


@pytest.mark.parametrize("backend", [NumpyBackend(), JaxBackend()], ids=["numpy", "jax"])
def test_get_graph_statistics(backend):
    e1 = HyperEdgeSet.from_dict(
        port_dict={"a": np.array([0, 1])},
        feature_dict={"x": np.array([1.0, 2.0])},
        backend=backend,
    )
    e2 = HyperEdgeSet.from_dict(
        port_dict={"a": np.array([0, 1])},
        feature_dict={"x": np.array([2.0, 4.0])},
        backend=backend,
    )
    g1 = Graph.from_dict(hyper_edge_set_dict={"T": e1}, n_addresses=2, backend=backend)
    g2 = Graph.from_dict(hyper_edge_set_dict={"T": e2}, n_addresses=2, backend=backend)
    stats = get_graph_statistics(g1, axis=None, norm_graph=g2)

    arr = np.array([1.0, 2.0])
    np.testing.assert_allclose(float(stats["T/x/rmse"]), np.sqrt(np.mean(arr**2)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/mae"]), np.mean(np.abs(arr)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/mean"]), np.mean(arr), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/std"]), np.std(arr), rtol=1e-5)
    assert "T/x/nrmse" in stats and "T/x/nmae" in stats


def test_get_series_statistics():
    stats = get_series_statistics([1.0, 2.0, np.nan, 4.0])

    arr = np.array([1.0, 2.0, np.nan, 4.0])
    np.testing.assert_allclose(stats["rmse"], np.sqrt(np.nanmean(arr**2)), rtol=1e-5)
    np.testing.assert_allclose(stats["mae"], np.nanmean(np.abs(arr)), rtol=1e-5)
    np.testing.assert_allclose(stats["mean"], np.nanmean(arr), rtol=1e-5)
    np.testing.assert_allclose(stats["std"], np.nanstd(arr), rtol=1e-5)
    np.testing.assert_allclose(stats["90th"], np.nanpercentile(arr, q=90), rtol=1e-5)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


def test_get_series_statistics_empty_or_all_nan():
    assert all(np.isnan(v) for v in get_series_statistics([]).values())
    assert all(np.isnan(v) for v in get_series_statistics([np.nan]).values())
