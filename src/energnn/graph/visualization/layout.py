# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Force-directed layout of a graph: where to put the nodes when no coordinate is given.

:func:`spring_layout` is a classic physics simulation where every node repels every other one and connected
nodes attract each other, which untangles the graph. It handles graphs of thousands of nodes in a few
seconds, by grouping the nodes that are far from each other (see :func:`_repulsion`).
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.spatial import cKDTree

#: Nodes closer than this fraction of the ideal distance repel each other as if they were that far apart, which
#: avoids infinite forces between nodes that landed on each other.
_MIN_DISTANCE = 0.1
#: Strength of the pull of every node toward the center of the drawing. It is too weak to deform a connected
#: graph, but it keeps the nodes that nothing holds (isolated nodes, separate components) from being pushed
#: away without end, which would shrink the rest of the drawing to a dot.
_GRAVITY = 1.0
#: Up to this many nodes the repulsion is computed pair by pair; beyond, the far nodes are grouped by cells.
_EXACT_UP_TO = 200
#: The cells that push a cell at one level of the grids, relative to it: 2 or 3 cells away along x or y. The
#: nearer cells are handled at a finer level, or node by node at the finest one (see :func:`_repulsion`).
_FAR_CELLS = np.array([(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if max(abs(dx), abs(dy)) > 1])


def spring_layout(n_nodes: int, edges: np.ndarray, *, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """
    Compute a Fruchterman-Reingold force-directed layout.

    The nodes start at random positions. At each iteration every node is pushed away from every other one
    (a repulsion decreasing with the distance) and pulled toward its neighbors (an attraction growing with
    the distance), and moves along the resulting force by at most a *temperature* that cools down to zero:
    large moves first to untangle, small ones at the end to settle. The result is centered and scaled into
    the ``[-1, 1]`` box.

    The cost of an iteration grows like the number of nodes and edges, not like the number of node pairs:
    beyond a few hundred nodes, the repulsion of the far nodes is computed by groups (see :func:`_repulsion`).

    :param n_nodes: Number of nodes to place.
    :param edges: Integer array of shape ``(n_edges, 2)`` listing node pairs.
    :param iterations: Number of relaxation steps.
    :param seed: Seed for the random initial positions.
    :return: Positions array of shape ``(n_nodes, 2)`` scaled to ``[-1, 1]``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n_nodes, 2))
    if n_nodes <= 1:
        return pos
    k = 1.0 / np.sqrt(n_nodes)  # the ideal distance between neighbors: nodes share the unit area
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    # two nodes attract each other once, however many edges join them, and a node does not attract itself
    links = np.sort(np.asarray(edges, dtype=int).reshape(-1, 2), axis=1)
    a, b = np.unique(links[links[:, 0] != links[:, 1]], axis=0).T
    depth = 0 if n_nodes <= _EXACT_UP_TO else int(np.log2(np.sqrt(n_nodes)))  # 1 to 4 nodes per cell of the finest grid
    for _ in range(iterations):
        displacement = k * k * _repulsion(pos, depth, _MIN_DISTANCE * k)  # repulsion for all pairs: k² / dist
        delta = pos[a] - pos[b]
        pull = delta * np.linalg.norm(delta, axis=-1, keepdims=True) / k  # attraction for neighbors: dist² / k
        displacement += _scatter_add(b, pull, n_nodes) - _scatter_add(a, pull, n_nodes)
        displacement -= _GRAVITY * (pos - pos.mean(axis=0))
        length = np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-9)
        pos += displacement / length * np.minimum(length, temperature)  # move along the force, capped
        temperature -= cooling
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    return pos / scale if scale > 0 else pos


def _scatter_add(index: np.ndarray, vectors: np.ndarray, n: int) -> np.ndarray:
    """Scatter-add 2-d ``vectors`` into ``n`` rows: row ``i`` of the result is the sum of the vectors whose ``index``
    is ``i``. It is what ``np.add.at(out, index, vectors)`` does, several times faster.

    :param index: The row of each vector, shape ``(m,)``.
    :param vectors: Shape ``(m, 2)``.
    :param n: Number of rows.
    :return: Shape ``(n, 2)``.
    """
    return np.stack([np.bincount(index, vectors[:, axis], minlength=n) for axis in (0, 1)], axis=-1)


@functools.lru_cache(maxsize=16)
def _pushed(g: int) -> list[np.ndarray]:
    """For each offset ``d`` of :data:`_FAR_CELLS`, the cells ``c`` of the ``g x g`` grid that the cell ``c + d``
    pushes at this level of :func:`_repulsion`: those whose parent is the parent of ``c + d`` or a neighbor of it.

    :return: One boolean mask of shape ``(g, g)`` per offset.
    """
    grid = np.indices((g, g))  # (2, g, g): the row and the column of every cell
    # halving a cell gives its parent; a cell outside the grid may pass, it has no mass and pushes nothing
    return [(np.abs((grid + d[:, None, None]) // 2 - grid // 2) <= 1).all(axis=0) for d in _FAR_CELLS]


def _repulsion(pos: np.ndarray, depth: int, min_distance: float) -> np.ndarray:
    """The repulsion of the spring layout: for each node, the sum over every other node of ``delta / dist²``,
    ``delta`` being the vector from the other node to this one (so the push decreases like ``1 / dist``).

    Doing that sum pair by pair costs ``n²`` operations per iteration, which is fine for a few hundred nodes
    (``depth = 0``) and hopeless for thousands. Beyond, the far nodes are grouped, following the idea of the
    Barnes-Hut algorithm: seen from far away, a group of nodes pushes like a single heavy node sitting at
    its center of mass. The bounding square of the nodes is cut into grids of 4 x 4, 8 x 8, ... up to
    ``2**depth x 2**depth`` cells, each cell being cut in four *children* at the next level. Then:

    - two nodes in the same cell or in adjacent cells of the finest grid are *near*: they repel each other
      one by one, exactly;
    - any other pair of nodes is handled once, at the coarsest level where their cells are not adjacent,
      by a push between the two cells: every node of one cell is pushed by the other cell as a whole. At
      each level, the cells that push a given cell are therefore those 2 or 3 cells away from it whose
      parent is its own parent or a neighbor of it (:func:`_pushed`): the cells further away were handled
      at a coarser level, through the parents.

    :param pos: The node positions, shape ``(n, 2)``.
    :param depth: The finest grid has ``2**depth`` cells per side; 0 means no grid, every pair is near.
    :param min_distance: Two nodes, or two cells, closer than this are taken to be this far apart.
    :return: The repulsion of each node, shape ``(n, 2)``, to be multiplied by ``k²``.
    """
    n = len(pos)
    low = pos.min(axis=0)
    side = float((pos.max(axis=0) - low).max()) or 1.0  # side of the bounding square
    finest = 2**depth
    cells = np.minimum(((pos - low) / side * finest).astype(int), finest - 1)  # (n, 2): cell of each node, finest grid
    force = np.zeros((n, 2))
    # far nodes, cell to cell, from the coarsest grid (4 x 4: in a 2 x 2 grid every cell is adjacent) to the finest
    for level in range(2, depth + 1):
        g = 2**level
        cell = cells >> (depth - level)  # cell of each node in the g x g grid: halving an index gives the parent
        index = cell[:, 0] * g + cell[:, 1]  # the cells numbered from 0 to g² - 1
        mass = np.bincount(index, minlength=g * g).astype(float)  # number of nodes per cell
        center = (_scatter_add(index, pos, g * g) / np.maximum(mass, 1.0)[:, None]).reshape(g, g, 2)  # center of mass
        # the grids are padded with 3 empty cells on each side, so that the cell c + d of every cell c is the
        # slice [3 + d, 3 + d + g] for every offset d: an empty cell has no mass and pushes nothing
        padded_center, padded_mass = np.pad(center, ((3, 3), (3, 3), (0, 0))), np.pad(mass.reshape(g, g), 3)
        push = np.zeros((g, g, 2))  # the push on one node of each cell
        for (dx, dy), pushed in zip(_FAR_CELLS, _pushed(g)):
            other = (slice(3 + dx, 3 + dx + g), slice(3 + dy, 3 + dy + g))
            delta = center - padded_center[other]
            dist2 = np.maximum((delta**2).sum(axis=-1), min_distance**2)
            push += delta * (pushed * padded_mass[other] / dist2)[..., None]
        force += push.reshape(-1, 2)[index]
    # near nodes, pair by pair: the k-d tree lists the pairs at most two cells apart along each axis (every pair
    # when there is no grid), among which the pairs of the same or adjacent cells are kept
    pairs = cKDTree(pos).query_pairs(2.0 * side / finest, p=np.inf, output_type="ndarray")
    i, j = pairs[(np.abs(cells[pairs[:, 0]] - cells[pairs[:, 1]]) <= 1).all(axis=1)].T
    delta = pos[i] - pos[j]
    dist2 = np.maximum((delta**2).sum(axis=-1), min_distance**2)
    push = delta / dist2[:, None]
    return force + _scatter_add(i, push, n) - _scatter_add(j, push, n)
