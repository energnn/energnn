# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.visualization.colors import resolve_colors
from energnn.graph.visualization.content import read_graph


@pytest.mark.parametrize("n_channels", [1, 2])
def test_address_colors_are_normalized_per_channel(mixed_order_graph, n_channels):
    raw = np.arange(4 * n_channels, dtype=float).reshape(4, n_channels) * 10.0 - 5.0
    scale = resolve_colors(read_graph(mixed_order_graph), address_colors=raw).addresses
    assert scale is not None and scale.n_channels == n_channels
    assert scale.channels.min() == 0.0 and scale.channels.max() == 1.0
    np.testing.assert_allclose(scale.low, raw.min(axis=0))
    np.testing.assert_allclose(scale.high, raw.max(axis=0))
    assert not scale.missing.any()


def test_constant_channel_and_missing_values(mixed_order_graph):
    scale = resolve_colors(read_graph(mixed_order_graph), address_colors=np.array([[1.0], [1.0], [np.nan], [1.0]])).addresses
    assert np.all(scale.channels == 0.5) and scale.missing.tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="all missing"):
        resolve_colors(read_graph(mixed_order_graph), address_colors=np.full((4, 1), np.nan))
    with pytest.raises(ValueError, match="address_colors"):
        resolve_colors(read_graph(mixed_order_graph), address_colors=np.zeros((4, 3)))


def test_hyper_edge_colors_share_one_scale_over_the_listed_classes(located_graph):
    topology = read_graph(located_graph)
    colors = resolve_colors(topology, hyper_edge_colors={"bus": ["load"], "line": ["flow"]})
    scale = colors.hyper_edges
    assert colors.addresses is None and scale is not None
    assert len(scale.channels) == len(topology.hyper_edges)  # one row per hyper-edge, whatever its class
    np.testing.assert_allclose([scale.low, scale.high], [[1.0], [10.0]])  # one range over buses and lines, NaN ignored
    by_key = {h.key: (scale.channels[row, 0], scale.missing[row]) for row, h in enumerate(topology.hyper_edges)}
    assert by_key[("bus", 0)] == (0.0, False) and by_key[("bus", 2)] == pytest.approx((2 / 9, False))
    assert by_key[("line", 0)] == (1.0, False) and by_key[("line", 1)] == (0.5, True)  # NaN: flagged, not averaged
    assert by_key[("theta", 0)] == (0.5, True)  # an unlisted class is "missing": it keeps its class color
    with pytest.raises(ValueError, match="1 or 2 feature names"):
        resolve_colors(topology, hyper_edge_colors={"bus": ["x", "y", "load"]})


def test_two_channels_for_hyper_edges_and_one_for_addresses(located_graph):
    colors = resolve_colors(
        read_graph(located_graph), address_colors=np.arange(4.0)[:, None], hyper_edge_colors={"bus": ["x", "load"]}
    )
    assert colors.hyper_edges.n_channels == 2 and colors.addresses.n_channels == 1
    np.testing.assert_allclose([colors.hyper_edges.low, colors.hyper_edges.high], [[0.0, 1.0], [4.0, 3.0]])
