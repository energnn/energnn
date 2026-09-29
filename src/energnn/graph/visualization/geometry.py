# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 4 of the pipeline: the drawable geometry of every hyper-edge, in layout units.

A stub is a short segment from its address to a marker; a pair is a curve between its two addresses,
bent by its rank among the parallel pairs so that they stay distinguishable; a self-loop is a marker
offset from its address with two fanned spokes; a hub (order 3+, or any placed object) is a marker with
one spoke per port, spokes to a repeated address being fanned out. Both renderers draw these polylines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energnn.graph.visualization.content import HyperEdge, ObjKey, Topology
from energnn.graph.visualization.positions import FAN_HEIGHT, STUB_LENGTH, Layout, address_radius, stub_direction

_CURVE_T = np.linspace(0.0, 1.0, 17)[:, None]
_LABEL_AT = 8  # index along a 17-point curve where its label sits (the middle)
_PAIR_LABELS_AT = (3, 13)  # where the two port labels of a pair sit, near each end


@dataclass(frozen=True)
class Geometry:
    """Drawable geometry of one hyper-edge."""

    lines: list[np.ndarray]  # polylines of shape (k, 2)
    marker: np.ndarray  # marker position, shape (2,)
    labels: list[np.ndarray]  # one label anchor per port


def geometries(topology: Topology, layout: Layout) -> dict[ObjKey, Geometry]:
    """The geometry of every hyper-edge that draws something (objects without ports draw nothing)."""
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
    """Rank the pairs sharing an address pair (across classes) for fanning, and the self-loops on an address."""
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
    t = _CURVE_T
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * control + t**2 * b


def fanned_curve(a: np.ndarray, b: np.ndarray, fan: float) -> np.ndarray:
    """A quadratic curve from ``a`` to ``b`` bent sideways by ``fan`` (0 is straight, ±1 the first pair of bulges)."""
    chord = b - a
    length = max(float(np.linalg.norm(chord)), 1e-9)
    normal = np.array([-chord[1], chord[0]]) / length
    height = fan * min(0.3 * length, FAN_HEIGHT)
    return _bezier(a, (a + b) / 2.0 + 2.0 * height * normal, b)


def _stub(anchor: np.ndarray, direction: np.ndarray, r_addr: float) -> Geometry:
    tip = anchor + STUB_LENGTH * r_addr * direction
    return Geometry([np.stack([anchor, tip])], tip, [(anchor + tip) / 2.0])


def loop_direction(rank: tuple[int, int]) -> np.ndarray:
    """Unit direction from an address to its ``rank``-th self-loop, loops spread around the address."""
    j, m = rank
    angle = 2.0 * np.pi * j / m + 0.6
    return np.array([np.cos(angle), np.sin(angle)])


def _loop(anchor: np.ndarray, rank: tuple[int, int], r_addr: float) -> Geometry:
    marker = anchor + STUB_LENGTH * r_addr * loop_direction(rank)
    curves = [fanned_curve(anchor, marker, -0.5), fanned_curve(anchor, marker, 0.5)]
    return Geometry(curves, marker, [curve[_LABEL_AT] for curve in curves])


def _pair(a: np.ndarray, b: np.ndarray, fan: float) -> Geometry:
    curve = fanned_curve(a, b, fan)
    return Geometry([curve], curve[_LABEL_AT], [curve[i] for i in _PAIR_LABELS_AT])


def _hub(hub: np.ndarray, addresses: np.ndarray, h: HyperEdge) -> Geometry:
    counts = {p: h.ports.count(p) for p in h.ports}
    seen: dict[int, int] = {}
    lines, labels = [], []
    for p in h.ports:
        j = seen.get(p, 0)
        seen[p] = j + 1
        if counts[p] == 1:
            lines.append(np.stack([hub, addresses[p]]))
            labels.append((hub + addresses[p]) / 2.0)
        else:  # spokes to the same address are fanned out
            curve = fanned_curve(hub, addresses[p], j - (counts[p] - 1) / 2.0)
            lines.append(curve)
            labels.append(curve[_LABEL_AT])
    return Geometry(lines, hub, labels)
