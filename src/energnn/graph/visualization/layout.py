# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Force-directed layout of a graph: where to put the nodes when no coordinate is given.

:func:`spring_layout` is a classic physics simulation where every node repels every other one and connected
nodes attract each other, which untangles the graph. The simulation runs in JAX, on the GPU when there is one,
and handles graphs of thousands of nodes in a few seconds on a CPU, by grouping the nodes that are far from
each other (see :func:`_repulsion`).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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
#: How many near nodes a node looks at in one pass over each row of three cells around it (see
#: :func:`_repulsion`): the finest grid has 1 to 4 nodes per cell, so one pass is usually enough.
_WINDOW = 48


def spring_layout(
    n_nodes: int, edges: np.ndarray, *, iterations: int = 150, seed: int = 0, fixed: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute a Fruchterman-Reingold force-directed layout.

    The nodes start at random positions. At each iteration every node is pushed away from every other one
    (a repulsion decreasing with the distance) and pulled toward its neighbors (an attraction growing with
    the distance), and moves along the resulting force by at most a *temperature* that cools down to zero:
    large moves first to untangle, small ones at the end to settle. The result is centered and scaled into
    the ``[-1, 1]`` box.

    Some nodes may be *fixed* at given positions: they push and pull the others without moving, which places
    the free nodes among them. The free nodes start from the fixed ones, each at the mean position of its
    neighbors placed before it, which is about where it ends up. The result is then left in the frame of the
    given positions, which should roughly fill the ``[-1, 1]`` box for the forces to be balanced.

    The cost of an iteration grows like the number of nodes and edges, not like the number of node pairs:
    beyond a few hundred nodes, the repulsion of the far nodes is computed by groups (see :func:`_repulsion`).
    The iterations run as one compiled JAX program (:func:`_relax`), compiled once per size of graph, the
    sizes being rounded up to powers of two.

    :param n_nodes: Number of nodes to place.
    :param edges: Integer array of shape ``(n_edges, 2)`` listing node pairs.
    :param iterations: Number of relaxation steps.
    :param seed: Seed for the random initial positions.
    :param fixed: Optional positions of shape ``(n_nodes, 2)``: the nodes whose row has no NaN stay there.
    :return: Positions array of shape ``(n_nodes, 2)``, scaled to ``[-1, 1]`` when no node is fixed.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n_nodes, 2))
    k = 1.0 / np.sqrt(max(n_nodes, 1))  # the ideal distance between neighbors: nodes share the unit area
    # two nodes attract each other once, however many edges join them, and a node does not attract itself
    links = np.sort(np.asarray(edges, dtype=int).reshape(-1, 2), axis=1)
    a, b = np.unique(links[links[:, 0] != links[:, 1]], axis=0).T
    known = np.zeros(n_nodes, dtype=bool)
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=float)
        known = np.isfinite(fixed).all(axis=1)
        pos[known] = fixed[known]
        # the free nodes start about where they will settle, wave after wave: the nodes joined to fixed ones at
        # the mean position of those, then the nodes joined to the ones just placed, and so on; the nodes that
        # no path joins to a fixed one keep their random position
        placed = known.copy()
        while True:
            at_front, from_back = ~placed[a] & placed[b], ~placed[b] & placed[a]
            free = np.concatenate([a[at_front], b[from_back]])
            if free.size == 0:
                break
            neighbor = np.concatenate([b[at_front], a[from_back]])
            hits = np.bincount(free, minlength=n_nodes)
            wave = hits > 0
            mean = _scatter_add(free, pos[neighbor], n_nodes)[wave] / hits[wave, None]
            pos[wave] = mean + rng.normal(scale=0.1 * k, size=mean.shape)  # a little apart from each other
            placed[wave] = True
    if n_nodes <= 1 or iterations <= 0:
        return pos
    depth = 0 if n_nodes <= _EXACT_UP_TO else int(np.log2(np.sqrt(n_nodes)))  # 1 to 4 nodes per cell of the finest grid
    # the arrays are padded to powers of two, so that a new size of graph seldom compiles the program again: the
    # padding nodes are fixed, weightless and out of every cell, and the padding edges join node 0 to itself
    size, n_edges = 1 << max(n_nodes - 1, 1).bit_length(), 1 << max(len(a) - 1, 0).bit_length()
    padded = np.zeros((size, 2))
    padded[:n_nodes] = pos
    real = np.arange(size) < n_nodes
    pos = _relax(
        jnp.asarray(padded, dtype=jnp.float32),
        jnp.asarray(np.pad(a, (0, n_edges - len(a)))),
        jnp.asarray(np.pad(b, (0, n_edges - len(b)))),
        jnp.asarray(np.pad(known, (0, size - n_nodes), constant_values=True)),
        jnp.asarray(real),
        jnp.asarray(iterations),
        jnp.asarray(k, dtype=jnp.float32),
        depth=depth,
    )
    pos = np.asarray(pos[:n_nodes], dtype=float)
    if known.any():
        return pos
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


@functools.partial(jax.jit, static_argnames="depth")
def _relax(
    pos: jax.Array,
    a: jax.Array,
    b: jax.Array,
    known: jax.Array,
    real: jax.Array,
    iterations: jax.Array,
    k: jax.Array,
    *,
    depth: int,
) -> jax.Array:
    """Run the iterations of :func:`spring_layout` on padded arrays.

    :param pos: The start positions, shape ``(size, 2)``.
    :param a: With ``b``, the pairs of nodes that attract each other, shape ``(n_edges,)`` each.
    :param known: Mask of the nodes that do not move, shape ``(size,)``.
    :param real: Mask of the nodes that are not padding, shape ``(size,)``.
    :param iterations: Number of relaxation steps.
    :param k: The ideal distance between neighbors.
    :param depth: The finest grid of the repulsion has ``2**depth`` cells per side; 0 means no grid.
    :return: The positions after the iterations, shape ``(size, 2)``.
    """
    n = len(pos)
    n_real = real.sum()
    temperature = 0.1
    cooling = temperature / (iterations + 1)

    def step(i: jax.Array, pos: jax.Array) -> jax.Array:
        displacement = k * k * _repulsion(pos, real, _MIN_DISTANCE * k, depth=depth)  # for all pairs: k² / dist
        delta = pos[a] - pos[b]
        pull = delta * jnp.linalg.norm(delta, axis=-1, keepdims=True) / k  # attraction for neighbors: dist² / k
        displacement += jax.ops.segment_sum(pull, b, n) - jax.ops.segment_sum(pull, a, n)
        center = jnp.sum(pos * real[:, None], axis=0) / n_real
        displacement -= _GRAVITY * (pos - center)
        length = jnp.maximum(jnp.linalg.norm(displacement, axis=-1, keepdims=True), 1e-9)
        displacement = jnp.where(known[:, None], 0.0, displacement)
        move = jnp.minimum(length, temperature - i * cooling)  # along the force, capped by the temperature
        return pos + displacement / length * move

    return lax.fori_loop(0, iterations, step, pos)


@functools.lru_cache(maxsize=16)
def _pushed(g: int) -> np.ndarray:
    """For each offset ``d`` of :data:`_FAR_CELLS`, the cells ``c`` of the ``g x g`` grid that the cell ``c + d``
    pushes at this level of :func:`_repulsion`: those whose parent is the parent of ``c + d`` or a neighbor of it.

    :return: Boolean masks of shape ``(len(_FAR_CELLS), g, g)``.
    """
    grid = np.indices((g, g))  # (2, g, g): the row and the column of every cell
    # halving a cell gives its parent; a cell outside the grid may pass, it has no mass and pushes nothing
    return np.stack([(np.abs((grid + d[:, None, None]) // 2 - grid // 2) <= 1).all(axis=0) for d in _FAR_CELLS])


@functools.partial(jax.jit, static_argnames="depth")
def _repulsion(pos: jax.Array, real: jax.Array, min_distance: jax.Array, *, depth: int) -> jax.Array:
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

    The near nodes are found without a tree, which a compiled program could not build: the nodes are sorted
    by cell, so that the nodes of a row of three adjacent cells follow each other, and every node looks at
    the nodes of each of its three rows, :data:`_WINDOW` at a time; it takes as many passes as the most
    crowded row needs, usually one.

    :param pos: The node positions, shape ``(n, 2)``.
    :param real: Mask of the nodes that count, shape ``(n,)``: the others have no mass and push nothing.
    :param min_distance: Two nodes, or two cells, closer than this are taken to be this far apart.
    :param depth: The finest grid has ``2**depth`` cells per side; 0 means no grid, every pair is near.
    :return: The repulsion of each node, shape ``(n, 2)``, to be multiplied by ``k²``.
    """
    n = len(pos)
    finest = 2**depth
    low = jnp.min(jnp.where(real[:, None], pos, jnp.inf), axis=0)
    side = jnp.maximum(jnp.max(jnp.where(real[:, None], pos, -jnp.inf) - low), 1e-9)  # side of the bounding square
    cells = jnp.clip(((pos - low) / side * finest).astype(jnp.int32), 0, finest - 1)  # (n, 2): cell at the finest grid
    force = jnp.zeros_like(pos)
    # far nodes, cell to cell, from the coarsest grid (4 x 4: in a 2 x 2 grid every cell is adjacent) to the finest
    for level in range(2, depth + 1):
        g = 2**level
        cell = cells >> (depth - level)  # cell of each node in the g x g grid: halving an index gives the parent
        index = jnp.where(real, cell[:, 0] * g + cell[:, 1], g * g)  # cells numbered from 0; the padding beyond
        mass = jax.ops.segment_sum(real.astype(pos.dtype), index, g * g + 1)[:-1]  # number of nodes per cell
        center = jax.ops.segment_sum(pos, index, g * g + 1)[:-1] / jnp.maximum(mass, 1.0)[:, None]  # center of mass
        center, mass = center.reshape(g, g, 2), mass.reshape(g, g)
        # the grids are padded with 3 empty cells on each side, so that the cell c + d of every cell c is the
        # slice [3 + d, 3 + d + g] for every offset d: an empty cell has no mass and pushes nothing
        padded_center, padded_mass = jnp.pad(center, ((3, 3), (3, 3), (0, 0))), jnp.pad(mass, 3)
        offsets = jnp.asarray(_FAR_CELLS + 3)
        other_center = jax.vmap(lambda d: lax.dynamic_slice(padded_center, (d[0], d[1], 0), (g, g, 2)))(offsets)
        other_mass = jax.vmap(lambda d: lax.dynamic_slice(padded_mass, (d[0], d[1]), (g, g)))(offsets)
        delta = center[None] - other_center  # (offsets, g, g, 2)
        dist2 = jnp.maximum((delta**2).sum(axis=-1), min_distance**2)
        push = (delta * (jnp.asarray(_pushed(g)) * other_mass / dist2)[..., None]).sum(axis=0)  # on one node per cell
        force += push.reshape(-1, 2)[jnp.minimum(index, g * g - 1)]
    # near nodes, one by one: sorted by cell, the nodes of three adjacent cells of a row follow each other
    index = jnp.where(real, cells[:, 0] * finest + cells[:, 1], finest * finest)
    order = jnp.argsort(index)
    rank = jnp.zeros(n, dtype=jnp.int32).at[order].set(jnp.arange(n, dtype=jnp.int32))  # place of each node in order
    numbers = jnp.arange(finest * finest)
    starts = jnp.searchsorted(index[order], numbers)  # first place of each cell in the order...
    ends = jnp.searchsorted(index[order], numbers, side="right")  # ... and the place after its last one
    rows = cells[:, :1] + jnp.array([-1, 0, 1])  # (n, 3): the row of each cell, and the rows on either side
    in_grid = (rows >= 0) & (rows < finest)
    rows = jnp.clip(rows, 0, finest - 1)
    first = starts[rows * finest + jnp.maximum(cells[:, 1:] - 1, 0)]  # (n, 3): the places of the three cells of a row
    last = ends[rows * finest + jnp.minimum(cells[:, 1:] + 1, finest - 1)]
    window = n if depth == 0 else _WINDOW
    last = jnp.where(in_grid, last, first)  # a row out of the grid holds nothing

    def look(chunk: jax.Array, force: jax.Array) -> jax.Array:
        places = first[..., None] + chunk * window + jnp.arange(window)  # (n, 3, window)
        seen = (places < last[..., None]) & (places != rank[:, None, None])  # not the node itself
        delta = pos[:, None, None] - pos[order[jnp.minimum(places, n - 1)]]
        dist2 = jnp.maximum((delta**2).sum(axis=-1), min_distance**2)
        return force + (delta * (seen / dist2)[..., None]).sum(axis=(1, 2))

    # one pass over the window of every node, and more when a row holds more nodes than the window
    return lax.fori_loop(0, (jnp.max(last - first) - 1) // window + 1, look, force)
