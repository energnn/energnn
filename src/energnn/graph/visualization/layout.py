# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Layout of a Graph in space: address positions, address colors and hyper-edge geometries.

Everything here is renderer-agnostic and depends on numpy only.

- Positions are always stored in 3D, with ``z = 0`` for 2D layouts, and always with a
  leading time axis: ``pos`` has shape ``(n_frames, n_nodes, 3)``. A single frame is the
  common case; a series of frames comes from ``positions`` given with an extra leading axis.
- Coordinates live in a ``[-1, 1]`` box, normalized once over all frames so that motion
  between frames is preserved.
- Address colors are 1, 2 or 3 channels per address, normalized per channel over all
  frames to ``[0, 1]``; the theme maps them to RGB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

# Object key: (hyper-edge class name, index among the real objects of that class).
ObjKey = tuple[str, int]


def spring_layout(n_nodes: int, edges: np.ndarray, *, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """
    Compute a Fruchterman-Reingold force-directed layout.

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
    k = 1.0 / np.sqrt(n_nodes)
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    if len(edges):
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 0.01)
        force = k * k / dist**2 - adjacency * dist / k
        displacement = (delta * force[..., None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-9)
        pos += displacement / length * np.minimum(length, temperature)
        temperature -= cooling
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    return pos / scale if scale > 0 else pos


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


class PlotData(NamedTuple):
    """Real (non-fictitious) content of a single Graph, laid out in space."""

    n_addr: int
    ndim: int  # 2 or 3
    classes: list[str]
    ports: dict[str, list[list[int]]]  # class -> per real object, port addresses (sorted port order)
    port_names: dict[str, list[str]]  # class -> sorted port names
    features: dict[str, list[dict[str, float]]]  # class -> per real object, feature name -> value
    pos: np.ndarray  # (n_frames, n_nodes, 3): addresses then hubs, z = 0 in 2D
    hub_ids: dict[ObjKey, int]  # (class, object index) -> hub row in pos
    colors: np.ndarray | None  # (n_frames, n_addr, C) normalized to [0, 1], or None
    color_range: np.ndarray | None  # (2, C): per-channel (min, max) of the raw values

    @property
    def n_frames(self) -> int:
        return self.pos.shape[0]


def _real_ports(hes) -> tuple[list[str], list[list[int]]]:
    """Sorted port names and, per real object, its port addresses in that order."""
    mask = np.asarray(hes.non_fictitious) > 0
    if hes.port_dict is None:
        return [], [[] for _ in range(int(mask.sum()))]
    names = sorted(hes.port_dict)
    stacked = np.stack([np.asarray(hes.port_dict[k]) for k in names], axis=-1)
    return names, [list(map(int, row)) for row in stacked[mask]]


def _real_features(hes, n_real: int) -> list[dict[str, float]]:
    """Per real object, feature name -> value."""
    if hes.feature_names is None:
        return [{} for _ in range(n_real)]
    feature_array = np.asarray(hes.feature_array)[np.asarray(hes.non_fictitious) > 0]
    names = sorted(hes.feature_names.items())
    return [{fn: float(feature_array[j, int(idx)]) for fn, idx in names} for j in range(len(feature_array))]


def _star_expansion(classes: list[str], ports: dict[str, list[list[int]]], n_addr: int):
    """Edges of the layout graph: order-2 objects link their ports, higher orders get a hub node."""
    layout_edges: list[tuple[int, int]] = []
    hub_ids: dict[ObjKey, int] = {}
    next_id = n_addr
    for name in classes:
        for i, edge_ports in enumerate(ports[name]):
            if len(edge_ports) == 2:
                layout_edges.append((edge_ports[0], edge_ports[1]))
            elif len(edge_ports) >= 3:
                hub_ids[(name, i)] = next_id
                layout_edges.extend((next_id, p) for p in edge_ports)
                next_id += 1
    return np.array(layout_edges, dtype=int).reshape(-1, 2), hub_ids, next_id


def _per_address_array(values: Any, name: str, address_mask: np.ndarray, n_addr: int, last_dims: tuple[int, ...]):
    """Validate a per-address array, drop fictitious rows and add the time axis.

    Accepts ``(n_addr, k)`` / ``(n_current, k)`` and ``(n_frames, n_addr, k)`` /
    ``(n_frames, n_current, k)`` for ``k`` in ``last_dims``; returns ``(n_frames, n_addr, k)``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] not in last_dims:
        raise ValueError(
            f"{name} must have shape (n_addresses, k) or (n_frames, n_addresses, k) with k in {last_dims}; "
            f"got {array.shape}."
        )
    if array.shape[1] == len(address_mask):
        array = array[:, address_mask]
    if array.shape[1] != n_addr:
        raise ValueError(f"{name} must have {n_addr} (real) or {len(address_mask)} (current) addresses; got {array.shape[1]}.")
    return array


def _normalize_positions(positions: Any, address_mask: np.ndarray, n_addr: int) -> np.ndarray:
    """User-given address coordinates -> ``(n_frames, n_addr, 3)`` fitted in the ``[-1, 1]`` box over all frames."""
    array = _per_address_array(positions, "positions", address_mask, n_addr, (2, 3))
    array = array - array.reshape(-1, array.shape[-1]).mean(axis=0)
    scale = np.abs(array).max()
    if scale > 0:
        array = array / scale
    if array.shape[-1] == 2:
        array = np.concatenate([array, np.zeros(array.shape[:-1] + (1,))], axis=-1)
    return array


def _normalize_colors(address_colors: Any, address_mask: np.ndarray, n_addr: int) -> tuple[np.ndarray, np.ndarray]:
    """User-given per-address channels -> ``(n_frames, n_addr, C)`` in ``[0, 1]`` and the raw ``(2, C)`` range."""
    array = _per_address_array(address_colors, "address_colors", address_mask, n_addr, (1, 2, 3))
    flat = array.reshape(-1, array.shape[-1])
    lo, hi = np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = np.where(hi > lo, (array - lo) / span, 0.5)
    return np.nan_to_num(normalized, nan=0.5), np.stack([lo, hi])


def extract_plot_data(
    graph: Graph, *, iterations: int, seed: int, positions: Any = None, address_colors: Any = None
) -> PlotData:
    """Collect real (non-fictitious) objects of a single Graph and lay them out.

    When ``positions`` is given, it provides the address coordinates, 2D or 3D, for one
    frame or a series of frames (real addresses, or padded length: fictitious rows are
    dropped); hyper-edge hubs are then placed at the barycenter of their ports instead of
    being laid out by the spring model. ``address_colors`` gives 1, 2 or 3 channels per
    address, with the same frame conventions.
    """
    if not graph.is_single:
        raise ValueError("plot_graph only handles single graphs; use separate_graphs() on a batch first.")

    g = graph.to_numpy_backend()
    address_mask = np.asarray(g.non_fictitious_addresses) > 0
    n_addr = int(address_mask.sum())

    classes = sorted(g.hyper_edge_sets)
    ports: dict[str, list[list[int]]] = {}
    port_names: dict[str, list[str]] = {}
    features: dict[str, list[dict[str, float]]] = {}
    for name in classes:
        port_names[name], ports[name] = _real_ports(g.hyper_edge_sets[name])
        features[name] = _real_features(g.hyper_edge_sets[name], len(ports[name]))

    layout_edges, hub_ids, n_nodes = _star_expansion(classes, ports, n_addr)
    if positions is None:
        flat = spring_layout(n_nodes, layout_edges, iterations=iterations, seed=seed)
        pos = np.concatenate([flat, np.zeros((n_nodes, 1))], axis=-1)[None]
        ndim = 2
    else:
        addr_pos = _normalize_positions(positions, address_mask, n_addr)
        ndim = 2 if np.all(addr_pos[..., 2] == 0) else 3
        pos = np.zeros((addr_pos.shape[0], n_nodes, 3))
        pos[:, :n_addr] = addr_pos
        for (name, i), hub_id in hub_ids.items():
            pos[:, hub_id] = addr_pos[:, ports[name][i]].mean(axis=1)

    colors = color_range = None
    if address_colors is not None:
        colors, color_range = _normalize_colors(address_colors, address_mask, n_addr)
        if colors.shape[0] != pos.shape[0]:
            if pos.shape[0] == 1:
                pos = np.repeat(pos, colors.shape[0], axis=0)
            elif colors.shape[0] == 1:
                colors = np.repeat(colors, pos.shape[0], axis=0)
            else:
                raise ValueError(f"positions have {pos.shape[0]} frames but address_colors have {colors.shape[0]}.")

    return PlotData(n_addr, ndim, classes, ports, port_names, features, pos, hub_ids, colors, color_range)


# ---------------------------------------------------------------------------
# Object geometries
# ---------------------------------------------------------------------------


class ObjGeom(NamedTuple):
    """Drawable geometry of one hyper-edge object, in layout coordinates (3D, z = 0 in 2D)."""

    lines: list[np.ndarray]  # polylines of shape (k, 3)
    marker: np.ndarray  # marker position, shape (3,)
    labels: list[np.ndarray]  # one label anchor per port


_BEZIER_T = np.linspace(0.0, 1.0, 17)[:, None]


def _bezier(a: np.ndarray, control: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = _BEZIER_T
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * control + t**2 * b


def _perpendicular(direction: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``direction``; in the xy-plane it is the usual left normal."""
    normal = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    if np.linalg.norm(normal) < 1e-9:  # direction is along z
        normal = np.array([1.0, 0.0, 0.0])
    return normal / np.linalg.norm(normal)


def _pair_ranks(data: PlotData) -> tuple[dict[ObjKey, float], dict[ObjKey, tuple[int, int]]]:
    """Rank order-2 objects sharing an address pair (across classes): fan offsets and self-loop ranks."""
    pair_groups: dict[tuple[int, int], list[ObjKey]] = {}
    for name in data.classes:
        for i, edge_ports in enumerate(data.ports[name]):
            if len(edge_ports) == 2:
                pair_groups.setdefault((min(edge_ports), max(edge_ports)), []).append((name, i))

    fan: dict[ObjKey, float] = {}
    loop_rank: dict[ObjKey, tuple[int, int]] = {}
    for (addr_a, addr_b), members in pair_groups.items():
        for j, member in enumerate(members):
            if addr_a == addr_b:
                loop_rank[member] = (j, len(members))
            else:
                fan[member] = j - (len(members) - 1) / 2.0
    return fan, loop_rank


def address_radius(n_addr: int) -> float:
    """Radius of the address circles in layout units (the ``[-1, 1]`` box), shrinking with their number.

    Both renderers draw addresses with this radius, so hyper-edge geometries can keep clear of them.
    """
    return float(np.clip(150.0 / np.sqrt(max(n_addr, 1)), 5.0, 13.0)) / 290.0


def stub_direction(class_index: int, i: int) -> np.ndarray:
    """Deterministic unit direction (in the xy-plane) of an order-1 stub, so several stubs stay visible."""
    angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _order1_geom(anchor: np.ndarray, class_index: int, i: int, r_addr: float) -> ObjGeom:
    """A short stub leaving the address; its marker sits clear of the address circle."""
    tip = anchor + 2.6 * r_addr * stub_direction(class_index, i)
    return ObjGeom([np.stack([anchor, tip])], tip, [(anchor + tip) / 2.0])


def loop_direction(rank: tuple[int, int]) -> np.ndarray:
    """Unit direction (in the xy-plane) from an address to its ``rank``-th self-loop."""
    j, m = rank
    angle = 2.0 * np.pi * j / m + 0.6
    return np.array([np.cos(angle), np.sin(angle), 0.0])


LOOP_RADIUS = 0.055


def _loop_geom(anchor: np.ndarray, rank: tuple[int, int], r_addr: float) -> ObjGeom:
    """A small circle just outside the address circle, in the xy-plane; several loops spread around it."""
    u = loop_direction(rank)
    v = np.array([-u[1], u[0], 0.0])
    r_loop = LOOP_RADIUS
    center = anchor + (r_addr + r_loop + 0.01) * u
    theta = np.linspace(0.0, 2.0 * np.pi, 25)[:, None]
    circle = center + r_loop * (np.cos(theta) * u + np.sin(theta) * v)
    labels = [
        center + 1.6 * r_loop * (np.cos(0.9) * u + np.sin(0.9) * v),
        center + 1.6 * r_loop * (np.cos(0.9) * u - np.sin(0.9) * v),
    ]
    return ObjGeom([circle], center + r_loop * u, labels)


def _pair_geom(a: np.ndarray, b: np.ndarray, fan: float) -> ObjGeom:
    """A Bezier curve between two addresses, bent according to its rank among parallel edges."""
    chord = b - a
    length = max(float(np.linalg.norm(chord)), 1e-9)
    normal = _perpendicular(chord / length)
    height = fan * min(0.3 * length, 0.09)
    curve = _bezier(a, (a + b) / 2.0 + 2.0 * height * normal, b)
    return ObjGeom([curve], curve[8], [curve[3], curve[13]])


def _hub_geom(hub: np.ndarray, port_pos: list[np.ndarray]) -> ObjGeom:
    """A hub marker with one spoke per port."""
    return ObjGeom([np.stack([hub, p]) for p in port_pos], hub, [(hub + p) / 2.0 for p in port_pos])


def object_geometries(data: PlotData, frame: int = 0) -> dict[ObjKey, ObjGeom]:
    """
    Geometry of every hyper-edge object at ``frame``, in layout coordinates.

    Order-2 edges sharing the same address pair (across all classes) are fanned out
    as symmetric Bezier curves so parallel edges stay distinguishable, and
    self-loops are drawn as small circles attached to their address.
    """
    pos = data.pos[frame]
    fan, loop_rank = _pair_ranks(data)
    r_addr = address_radius(data.n_addr)

    geoms: dict[ObjKey, ObjGeom] = {}
    for class_index, name in enumerate(data.classes):
        for i, edge_ports in enumerate(data.ports[name]):
            key = (name, i)
            if len(edge_ports) == 1:
                geoms[key] = _order1_geom(pos[edge_ports[0]], class_index, i, r_addr)
            elif key in loop_rank:
                geoms[key] = _loop_geom(pos[edge_ports[0]], loop_rank[key], r_addr)
            elif len(edge_ports) == 2:
                geoms[key] = _pair_geom(pos[edge_ports[0]], pos[edge_ports[1]], fan[key])
            elif len(edge_ports) >= 3:
                geoms[key] = _hub_geom(pos[data.hub_ids[key]], [pos[p] for p in edge_ports])
    return geoms


def object_descriptors(data: PlotData) -> dict[str, list[dict[str, Any]]]:
    """Topology-only description of every object, for renderers that compute geometry themselves.

    Per class, one dict per real object with its ``ports``, its ``kind`` (``stub``, ``loop``,
    ``pair``, ``hub`` or ``none`` for port-less objects) and the kind's parameters: the stub
    ``direction``, the loop ``direction``, the pair ``fan`` offset, or the ``hub`` row of ``pos``.
    """
    fan, loop_rank = _pair_ranks(data)
    out: dict[str, list[dict[str, Any]]] = {}
    for class_index, name in enumerate(data.classes):
        items = []
        for i, edge_ports in enumerate(data.ports[name]):
            key = (name, i)
            item: dict[str, Any] = {"ports": edge_ports}
            if len(edge_ports) == 1:
                item |= {"kind": "stub", "direction": stub_direction(class_index, i)[:2].round(6).tolist()}
            elif key in loop_rank:
                item |= {"kind": "loop", "direction": loop_direction(loop_rank[key])[:2].round(6).tolist()}
            elif len(edge_ports) == 2:
                item |= {"kind": "pair", "fan": fan[key]}
            elif len(edge_ports) >= 3:
                item |= {"kind": "hub", "hub": data.hub_ids[key]}
            else:
                item |= {"kind": "none"}
            items.append(item)
        out[name] = items
    return out
