# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 4 of the drawing pipeline: the drawable geometry of every hyper-edge.

The addresses are circles at the positions of the :class:`~.positions.Layout`; this module says what to
draw for each object, in layout units, according to its :attr:`~.content.HyperEdge.kind`:

- a *stub* (order 1) is a short segment leaving its address, with the class marker at the tip;
- a *pair* (order 2) is a curve between its two addresses, with the marker in the middle; when several
  pairs join the same two addresses (a multi-graph), they are fanned out into curves of increasing bulge
  on either side of the straight line, so that each one stays visible and hoverable;
- a *loop* (order 2 on one address) is a marker beside the address with two spokes, fanned out so that
  they do not overlap; several loops on one address are spread around it;
- a *hub* (order 3 or more) and a *placed* object (any order) are a marker at the hub position with one
  spoke per port; spokes going to the same address are fanned out like parallel pairs.

Every drawing is a :class:`Geometry`: polylines, a marker position and one label anchor per port (where
the port name is written on hover or with ``port_labels``). Both renderers only draw these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energnn.graph.visualization.content import HyperEdge, ObjKey, Topology
from energnn.graph.visualization.positions import FAN_HEIGHT, STUB_LENGTH, Layout, address_radius, stub_direction

# Curves are sampled at 17 points, which is enough for a smooth quadratic curve at any zoom.
_CURVE_T = np.linspace(0.0, 1.0, 17)[:, None]
_LABEL_AT = 8  # the middle sample of a 17-point curve, where its marker or label sits
_PAIR_LABELS_AT = (3, 13)  # the samples near each end of a pair, where its two port labels sit


@dataclass(frozen=True)
class Geometry:
    """The drawable geometry of one hyper-edge, in layout units.

    :param lines: The polylines to stroke, each of shape ``(k, 2)`` (2 points for a segment, 17 for a curve).
    :param marker: Where the class marker sits, shape ``(2,)``.
    :param labels: One anchor per port, in port order, where the port name is written.
    """

    lines: list[np.ndarray]
    marker: np.ndarray
    labels: list[np.ndarray]


def geometries(topology: Topology, layout: Layout) -> dict[ObjKey, Geometry]:
    """The geometry of every object that draws something.

    :return: ``{object key: geometry}``; objects of kind ``"none"`` (no port) are absent.
    """
    r_addr = address_radius(topology.n_addresses)
    fan, loop_rank = _pair_ranks(topology)
    out: dict[ObjKey, Geometry] = {}
    for class_index, cls in enumerate(topology.classes):
        for h in topology.of(cls):
            pos = layout.addresses
            if h.kind == "stub":
                out[h.key] = _stub(pos[h.ports[0]], stub_direction(class_index, h.index), r_addr)
            elif h.kind == "loop":
                out[h.key] = _loop(pos[h.ports[0]], loop_rank[h.key], r_addr)
            elif h.kind == "pair":
                out[h.key] = _pair(pos[h.ports[0]], pos[h.ports[1]], fan[h.key])
            elif h.kind in ("hub", "placed"):
                out[h.key] = _hub(layout.hubs[h.key], pos, h)
    return out


def _pair_ranks(topology: Topology) -> tuple[dict[ObjKey, float], dict[ObjKey, tuple[int, int]]]:
    """Rank the objects that would otherwise overlap: parallel pairs and multiple loops.

    Pairs joining the same two addresses (whatever their classes) get fan offsets centered on zero, e.g.
    ``-1, 0, 1`` for three of them: the middle one is straight, the others bulge on either side. Loops on
    the same address get ``(rank, count)``, used to spread them around the address.

    :return: ``{pair key: fan offset}`` and ``{loop key: (rank, count)}``.
    """
    groups: dict[tuple[int, int], list[ObjKey]] = {}
    loops: dict[int, list[ObjKey]] = {}
    for h in topology.hyper_edges:
        if h.kind == "pair":
            groups.setdefault((min(h.ports), max(h.ports)), []).append(h.key)
        elif h.kind == "loop":
            loops.setdefault(h.ports[0], []).append(h.key)
    fan = {key: j - (len(keys) - 1) / 2.0 for keys in groups.values() for j, key in enumerate(keys)}
    loop_rank = {key: (j, len(keys)) for keys in loops.values() for j, key in enumerate(keys)}
    return fan, loop_rank


def _bezier(a: np.ndarray, control: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sample the quadratic Bezier curve from ``a`` to ``b`` pulled toward ``control``: 17 points of shape (17, 2)."""
    t = _CURVE_T
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * control + t**2 * b


def fanned_curve(a: np.ndarray, b: np.ndarray, fan: float) -> np.ndarray:
    """A curve from ``a`` to ``b`` bent sideways according to its fan offset.

    ``fan = 0`` gives the straight segment (sampled as a curve). ``fan = ±1`` bends it one *fan height* on
    either side, ``±2`` twice as much, and so on. The bulge is capped at 30 % of the chord length so that
    short connections do not turn into blobs. The control point of the Bezier curve is put at twice the
    wanted bulge, because a quadratic curve only reaches halfway to its control point.

    :param a: Start point, shape ``(2,)``.
    :param b: End point, shape ``(2,)``.
    :param fan: The fan offset.
    :return: 17 points of shape ``(17, 2)``.
    """
    chord = b - a
    length = max(float(np.linalg.norm(chord)), 1e-9)
    normal = np.array([-chord[1], chord[0]]) / length  # unit vector perpendicular to the chord
    height = fan * min(0.3 * length, FAN_HEIGHT)
    return _bezier(a, (a + b) / 2.0 + 2.0 * height * normal, b)


def _stub(anchor: np.ndarray, direction: np.ndarray, r_addr: float) -> Geometry:
    """A segment from the address ``anchor`` along ``direction``, the marker at its tip, the label halfway."""
    tip = anchor + STUB_LENGTH * r_addr * direction
    return Geometry([np.stack([anchor, tip])], tip, [(anchor + tip) / 2.0])


def loop_direction(rank: tuple[int, int]) -> np.ndarray:
    """Unit direction from an address to its ``rank``-th loop out of ``count``: the loops are spread evenly
    around the address, starting at a fixed angle so that a single loop does not hide behind a stub."""
    j, m = rank
    angle = 2.0 * np.pi * j / m + 0.6
    return np.array([np.cos(angle), np.sin(angle)])


def _loop(anchor: np.ndarray, rank: tuple[int, int], r_addr: float) -> Geometry:
    """A self-loop: the marker a stub length away from the address, two spokes fanned on either side of the
    straight line (both start at the address center, so the loop stays attached), one label on each spoke."""
    marker = anchor + STUB_LENGTH * r_addr * loop_direction(rank)
    curves = [fanned_curve(anchor, marker, -0.5), fanned_curve(anchor, marker, 0.5)]
    return Geometry(curves, marker, [curve[_LABEL_AT] for curve in curves])


def _pair(a: np.ndarray, b: np.ndarray, fan: float) -> Geometry:
    """A curve between two addresses, the marker in its middle, one label near each end (port order: a, b)."""
    curve = fanned_curve(a, b, fan)
    return Geometry([curve], curve[_LABEL_AT], [curve[i] for i in _PAIR_LABELS_AT])


def _hub(hub: np.ndarray, addresses: np.ndarray, h: HyperEdge) -> Geometry:
    """A marker at ``hub`` with one spoke per port, in port order.

    A spoke to an address hit by a single port is a straight segment; when several ports of the object hit
    the same address, their spokes are fanned out like parallel pairs, so that every port keeps its own line
    and label.
    """
    counts = {p: h.ports.count(p) for p in h.ports}
    seen: dict[int, int] = {}  # how many spokes to each address were drawn so far: the rank of the next one
    lines, labels = [], []
    for p in h.ports:
        j = seen.get(p, 0)
        seen[p] = j + 1
        if counts[p] == 1:
            lines.append(np.stack([hub, addresses[p]]))
            labels.append((hub + addresses[p]) / 2.0)
        else:
            curve = fanned_curve(hub, addresses[p], j - (counts[p] - 1) / 2.0)
            lines.append(curve)
            labels.append(curve[_LABEL_AT])
    return Geometry(lines, hub, labels)
