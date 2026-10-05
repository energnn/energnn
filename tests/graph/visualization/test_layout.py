# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.visualization.layout import _repulsion, spring_layout


def test_spring_layout_shape_and_scale():
    pos = spring_layout(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    assert pos.shape == (5, 2) and np.abs(pos).max() <= 1.0 + 1e-6
    assert spring_layout(3, np.zeros((0, 2), dtype=int)).shape == (3, 2)


def test_spring_layout_ignores_repeated_edges_and_self_loops():
    chain = np.array([[0, 1], [1, 2], [2, 3]])
    repeated = np.array([[0, 1], [1, 0], [0, 1], [1, 2], [2, 2], [3, 2]])  # same links, given several times and reversed
    np.testing.assert_allclose(spring_layout(4, repeated), spring_layout(4, chain), atol=1e-9)


def _clustered_points(n: int) -> np.ndarray:
    """Half of the points in a dense blob, half spread over the box: what a layout looks like while it untangles."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(scale=0.05, size=(n // 2, 2)), rng.uniform(-1.0, 1.0, size=(n - n // 2, 2))])


def test_repulsion_without_grid_is_the_sum_over_every_pair():
    pos = _clustered_points(60)
    delta = pos[:, None, :] - pos[None, :, :]
    dist2 = np.maximum((delta**2).sum(axis=-1), 0.01**2)
    np.fill_diagonal(dist2, np.inf)  # a node does not repel itself
    np.testing.assert_allclose(_repulsion(pos, 0, 0.01), (delta / dist2[..., None]).sum(axis=1), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("depth", [2, 4, 6])
def test_grouped_repulsion_approximates_the_sum_over_every_pair(depth):
    pos = _clustered_points(1500)
    exact, grouped = _repulsion(pos, 0, 1e-3), _repulsion(pos, depth, 1e-3)
    error = np.linalg.norm(grouped - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert np.median(error) < 0.1 and np.quantile(error, 0.95) < 0.25
    assert np.abs(grouped.sum(axis=0)).max() < 1e-6 * np.abs(grouped).sum()  # pushes cancel out: no drift of the whole


def test_spring_layout_scales_to_thousands_of_nodes():
    n = 3000  # a ring with chords, big enough for the grids of the grouped repulsion to be several levels deep
    ring = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    chords = np.stack([np.arange(0, n, 50), (np.arange(0, n, 50) + 7) % n], axis=1)
    pos = spring_layout(n, np.concatenate([ring, chords]), iterations=20)
    assert pos.shape == (n, 2) and np.isfinite(pos).all() and np.abs(pos).max() <= 1.0 + 1e-6


def test_spring_layout_keeps_what_is_not_connected_together():
    n = 400  # a ring, and 200 nodes that nothing holds
    ring = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    pos = spring_layout(n + 200, ring)
    # were the isolated nodes pushed away without end, the ring would only fill a small part of the box
    assert np.ptp(pos[:n], axis=0).min() > 1.0
