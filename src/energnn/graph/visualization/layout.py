# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Layout of a Graph in the plane: address positions and hyper-edge geometries.

Everything here is renderer-agnostic and depends on numpy only. Coordinates live in a
``[-1, 1]`` box so that marker sizes and offsets are consistent across renderers.
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
# Content extraction and address placement
# ---------------------------------------------------------------------------


class PlotData(NamedTuple):
    """Real (non-fictitious) content of a single Graph, laid out in the plane."""

    n_addr: int
    classes: list[str]
    ports: dict[str, list[list[int]]]  # class -> per real object, port addresses (sorted port order)
    port_names: dict[str, list[str]]  # class -> sorted port names
    features: dict[str, list[dict[str, float]]]  # class -> per real object, feature name -> value
    pos: np.ndarray  # star-expansion layout: addresses then hubs
    hub_ids: dict[ObjKey, int]  # (class, object index) -> hub row in pos


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


def _normalize_positions(positions: Any, address_mask: np.ndarray, n_addr: int) -> np.ndarray:
    """Validate user-given address coordinates and fit them in the ``[-1, 1]`` layout box."""
    addr_pos = np.asarray(positions, dtype=float)
    if addr_pos.ndim == 2 and addr_pos.shape[0] == len(address_mask):
        addr_pos = addr_pos[address_mask]
    if addr_pos.shape != (n_addr, 2):
        raise ValueError(f"positions must have shape ({n_addr}, 2) or (current addresses, 2); got {addr_pos.shape}.")
    addr_pos = addr_pos - addr_pos.mean(axis=0)
    scale = np.abs(addr_pos).max()
    return addr_pos / scale if scale > 0 else addr_pos


def extract_plot_data(graph: Graph, *, iterations: int, seed: int, positions: Any = None) -> PlotData:
    """Collect real (non-fictitious) objects of a single Graph and lay them out.

    When ``positions`` is given, it provides the address coordinates (real addresses,
    or padded length — fictitious rows are dropped); hyper-edge hubs are then placed
    at the barycenter of their ports instead of being laid out by the spring model.
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
        pos = spring_layout(n_nodes, layout_edges, iterations=iterations, seed=seed)
    else:
        pos = np.zeros((n_nodes, 2))
        pos[:n_addr] = _normalize_positions(positions, address_mask, n_addr)
        for (name, i), hub_id in hub_ids.items():
            pos[hub_id] = pos[ports[name][i]].mean(axis=0)

    return PlotData(n_addr, classes, ports, port_names, features, pos, hub_ids)


# ---------------------------------------------------------------------------
# Object geometries
# ---------------------------------------------------------------------------


class ObjGeom(NamedTuple):
    """Drawable geometry of one hyper-edge object, in layout coordinates."""

    lines: list[np.ndarray]  # polylines of shape (k, 2)
    marker: np.ndarray  # marker position, shape (2,)
    labels: list[np.ndarray]  # one label anchor per port


_BEZIER_T = np.linspace(0.0, 1.0, 17)[:, None]


def _bezier(a: np.ndarray, control: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = _BEZIER_T
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * control + t**2 * b


def _rotate(u: np.ndarray, phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


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


def _order1_geom(anchor: np.ndarray, class_index: int, i: int) -> ObjGeom:
    """A short stub leaving the address, with a deterministic angle so several stubs stay visible."""
    angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
    tip = anchor + 0.05 * np.array([np.cos(angle), np.sin(angle)])
    return ObjGeom([np.stack([anchor, tip])], tip, [(anchor + tip) / 2.0])


def _loop_geom(anchor: np.ndarray, rank: tuple[int, int]) -> ObjGeom:
    """A small circle beside the address; several loops spread around it."""
    j, m = rank
    u = np.array([np.cos(2.0 * np.pi * j / m + 0.6), np.sin(2.0 * np.pi * j / m + 0.6)])
    r_loop = 0.055
    center = anchor + 1.7 * r_loop * u
    theta = np.linspace(0.0, 2.0 * np.pi, 25)[:, None]
    circle = center + r_loop * np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    labels = [center + 1.6 * r_loop * _rotate(u, 0.9), center + 1.6 * r_loop * _rotate(u, -0.9)]
    return ObjGeom([circle], center + r_loop * u, labels)


def _pair_geom(a: np.ndarray, b: np.ndarray, fan: float) -> ObjGeom:
    """A Bezier curve between two addresses, bent according to its rank among parallel edges."""
    chord = b - a
    length = max(float(np.linalg.norm(chord)), 1e-9)
    normal = np.array([-chord[1], chord[0]]) / length
    height = fan * min(0.3 * length, 0.09)
    curve = _bezier(a, (a + b) / 2.0 + 2.0 * height * normal, b)
    return ObjGeom([curve], curve[8], [curve[3], curve[13]])


def _hub_geom(hub: np.ndarray, port_pos: list[np.ndarray]) -> ObjGeom:
    """A hub marker with one spoke per port."""
    return ObjGeom([np.stack([hub, p]) for p in port_pos], hub, [(hub + p) / 2.0 for p in port_pos])


def object_geometries(data: PlotData) -> dict[ObjKey, ObjGeom]:
    """
    Geometry of every hyper-edge object, in layout coordinates.

    Order-2 edges sharing the same address pair (across all classes) are fanned out
    as symmetric Bezier curves so parallel edges stay distinguishable, and
    self-loops are drawn as small circles attached to their address.
    """
    pos = data.pos
    fan, loop_rank = _pair_ranks(data)

    geoms: dict[ObjKey, ObjGeom] = {}
    for class_index, name in enumerate(data.classes):
        for i, edge_ports in enumerate(data.ports[name]):
            key = (name, i)
            if len(edge_ports) == 1:
                geoms[key] = _order1_geom(pos[edge_ports[0]], class_index, i)
            elif key in loop_rank:
                geoms[key] = _loop_geom(pos[edge_ports[0]], loop_rank[key])
            elif len(edge_ports) == 2:
                geoms[key] = _pair_geom(pos[edge_ports[0]], pos[edge_ports[1]], fan[key])
            elif len(edge_ports) >= 3:
                geoms[key] = _hub_geom(pos[data.hub_ids[key]], [pos[p] for p in edge_ports])
    return geoms
