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
    margin: float  # how far geometries (stubs, loops, curves, markers) may reach beyond the [-1, 1] box
    inferred: np.ndarray  # (n_frames, n_addr) bool: position not given (NaN) and reconstructed from the graph
    missing_colors: np.ndarray | None  # (n_frames, n_addr) bool: color not given (NaN), address left hollow

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
                layout_edges.extend((next_id, p) for p in sorted(set(edge_ports)))
                next_id += 1
    return np.array(layout_edges, dtype=int).reshape(-1, 2), hub_ids, next_id


def _place_hubs(
    pos: np.ndarray, hub_ids: dict[ObjKey, int], ports: dict[str, list[list[int]]], classes: list[str], from_layout: bool
) -> None:
    """Position the hub of every order-3+ object, for all frames, in place.

    A hub sits at the barycenter of its *distinct* port addresses (kept from the spring layout
    when ``from_layout``). Degenerate cases are handled for any order: a hub whose ports all hit
    one address is offset from it like an order-1 stub; a hub landing on one of its addresses is
    pushed away the same way; hubs sharing the same address set are spread side by side.
    """
    n_addr = min(hub_ids.values()) if hub_ids else pos.shape[1]  # hub rows come after the addresses
    r_addr = address_radius(n_addr)
    by_addresses: dict[tuple[int, ...], list[ObjKey]] = {}
    for key in hub_ids:
        by_addresses.setdefault(tuple(sorted(set(ports[key[0]][key[1]]))), []).append(key)
    for addresses, keys in by_addresses.items():
        anchors = pos[:, list(addresses)]  # (n_frames, n_distinct, 3)
        for j, (name, i) in enumerate(keys):
            direction = stub_direction(classes.index(name), i)
            if len(addresses) == 1:
                hub = anchors[:, 0] + STUB_LENGTH * r_addr * direction
            elif from_layout:
                hub = pos[:, hub_ids[(name, i)]].copy()
            else:
                hub = anchors.mean(axis=1)
                if len(keys) > 1:  # parallel hubs: spread them across the first spoke
                    chord = anchors[:, 1] - anchors[:, 0]
                    normal = np.stack([-chord[:, 1], chord[:, 0], np.zeros(len(chord))], axis=1)
                    norms = np.linalg.norm(normal, axis=1, keepdims=True)
                    normal = np.where(norms > 1e-9, normal / np.maximum(norms, 1e-9), np.array([1.0, 0.0, 0.0]))
                    hub = hub + (j - (len(keys) - 1) / 2.0) * 2.0 * r_addr * normal
            too_close = np.linalg.norm(anchors - hub[:, None], axis=-1).min(axis=1) < 1.5 * r_addr
            hub = np.where(too_close[:, None], hub + STUB_LENGTH * r_addr * direction, hub)
            pos[:, hub_ids[(name, i)]] = hub


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


def _adjacency(n_addr: int, ports: dict[str, list[list[int]]]):
    """Sparse symmetric adjacency between addresses: the ports of one object are pairwise neighbors."""
    import scipy.sparse  # type: ignore[import-untyped]

    rows, cols = [], []
    for edge_ports in (p for plist in ports.values() for p in plist):
        distinct = sorted(set(edge_ports))
        for a in distinct:
            for b in distinct:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    weights = np.ones(len(rows))
    return scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(n_addr, n_addr)).tocsr()


def _fill_missing_positions(array: np.ndarray, known: np.ndarray, adjacency, seed: int) -> None:
    """Replace NaN address coordinates in place, frame by frame.

    Unknown addresses of a component holding at least one known address get the harmonic
    (Tutte) embedding: each sits at the mean of its neighbors, solved as one sparse linear
    system with the known addresses as boundary. Components without any known address get a
    small spring layout placed beside the known cloud.
    """
    import scipy.sparse  # type: ignore[import-untyped]
    import scipy.sparse.csgraph  # type: ignore[import-untyped]
    import scipy.sparse.linalg  # type: ignore[import-untyped]

    d = array.shape[2]
    _, labels = scipy.sparse.csgraph.connected_components(adjacency, directed=False)
    laplacian = scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    laplacian = laplacian.tocsr()
    for t in range(array.shape[0]):
        known_t = known[t]
        if known_t.all():
            continue
        known_pos = array[t, known_t]
        center = known_pos.mean(axis=0)
        extent = max(float(np.abs(known_pos - center).max()), 1e-9)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            unknown = members[~known_t[members]]
            if unknown.size == 0:
                continue
            boundary = members[known_t[members]]
            if boundary.size:
                rhs = -laplacian[unknown][:, boundary] @ array[t, boundary]
                array[t, unknown] = scipy.sparse.linalg.spsolve(laplacian[unknown][:, unknown].tocsc(), rhs).reshape(-1, d)
            else:  # nothing known in this component: a small spring layout to the right of the known cloud
                local = adjacency[members][:, members].tocoo()
                edges = np.stack([local.row, local.col], axis=1) if local.nnz else np.zeros((0, 2), dtype=int)
                flat = spring_layout(members.size, edges, iterations=60, seed=seed)[:, :d]
                if d == 3:
                    flat = np.concatenate([flat, np.zeros((members.size, 1))], axis=1)
                offset = center.copy()
                offset[0] += 1.5 * extent
                array[t, members] = offset + 0.35 * extent * flat


def _normalize_positions(
    positions: Any, address_mask: np.ndarray, n_addr: int, ports: dict[str, list[list[int]]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """User-given address coordinates -> ``(n_frames, n_addr, 3)`` fitted in the ``[-1, 1]`` box over all frames.

    NaN rows are inferred from the graph (see :func:`_fill_missing_positions`); the returned
    ``(n_frames, n_addr)`` mask tells which addresses were inferred.
    """
    array = _per_address_array(positions, "positions", address_mask, n_addr, (2, 3))
    known = ~np.isnan(array).any(axis=-1)
    if not known.any():
        raise ValueError("positions are all missing (NaN); give at least one address coordinate, or no positions at all.")
    if not known.all():
        _fill_missing_positions(array, known, _adjacency(n_addr, ports), seed)
    array = array - array.reshape(-1, array.shape[-1]).mean(axis=0)
    scale = np.abs(array).max()
    if scale > 0:
        array = array / scale
    if array.shape[-1] == 2:
        array = np.concatenate([array, np.zeros(array.shape[:-1] + (1,))], axis=-1)
    return array, ~known


def _normalize_colors(address_colors: Any, address_mask: np.ndarray, n_addr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """User-given per-address channels -> ``(n_frames, n_addr, C)`` in ``[0, 1]``, the raw ``(2, C)`` range, and the
    ``(n_frames, n_addr)`` mask of addresses with a missing (NaN) color."""
    array = _per_address_array(address_colors, "address_colors", address_mask, n_addr, (1, 2, 3))
    missing = np.isnan(array).any(axis=-1)
    if missing.all():
        raise ValueError("address_colors are all missing (NaN).")
    flat = array.reshape(-1, array.shape[-1])
    lo, hi = np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = np.where(hi > lo, (array - lo) / span, 0.5)
    return np.nan_to_num(normalized, nan=0.5), np.stack([lo, hi]), missing


def extract_plot_data(
    graph: Graph, *, iterations: int, seed: int, positions: Any = None, address_colors: Any = None
) -> PlotData:
    """Collect real (non-fictitious) objects of a single Graph and lay them out.

    When ``positions`` is given, it provides the address coordinates, 2D or 3D, for one
    frame or a series of frames (real addresses, or padded length: fictitious rows are
    dropped); hyper-edge hubs are then placed at the barycenter of their ports instead of
    being laid out by the spring model; NaN rows are reconstructed from the graph (harmonic
    embedding between the known addresses). ``address_colors`` gives 1, 2 or 3 channels per
    address, with the same frame conventions; a NaN leaves the address uncolored.
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
        inferred = np.zeros((1, n_addr), dtype=bool)
    else:
        addr_pos, inferred = _normalize_positions(positions, address_mask, n_addr, ports, seed)
        ndim = 2 if np.all(addr_pos[..., 2] == 0) else 3
        pos = np.zeros((addr_pos.shape[0], n_nodes, 3))
        pos[:, :n_addr] = addr_pos
    _place_hubs(pos, hub_ids, ports, classes, from_layout=positions is None)

    colors = color_range = missing_colors = None
    if address_colors is not None:
        colors, color_range, missing_colors = _normalize_colors(address_colors, address_mask, n_addr)
        if colors.shape[0] != pos.shape[0]:
            if pos.shape[0] == 1:
                pos = np.repeat(pos, colors.shape[0], axis=0)
                inferred = np.repeat(inferred, colors.shape[0], axis=0)
            elif colors.shape[0] == 1:
                colors = np.repeat(colors, pos.shape[0], axis=0)
                missing_colors = np.repeat(missing_colors, pos.shape[0], axis=0)
            else:
                raise ValueError(f"positions have {pos.shape[0]} frames but address_colors have {colors.shape[0]}.")

    margin = layout_margin(n_addr, ports)
    return PlotData(
        n_addr, ndim, classes, ports, port_names, features, pos, hub_ids, colors, color_range, margin, inferred, missing_colors
    )


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


STUB_LENGTH = 2.6  # in address radii
FAN_HEIGHT = 0.09  # largest bulge of a fanned-out parallel edge


def layout_margin(n_addr: int, ports: dict[str, list[list[int]]]) -> float:
    """Room to keep around the ``[-1, 1]`` box so that stubs, self-loops and fanned edges stay in view."""
    r_addr = address_radius(n_addr)
    reach = 0.62 * r_addr  # a class marker centered on an address
    pairs: dict[tuple[int, int], int] = {}
    fanned = False
    for edge_ports in (p for plist in ports.values() for p in plist):
        if len(edge_ports) == 1 or (len(edge_ports) >= 3 and len(set(edge_ports)) == 1):
            reach = max(reach, (STUB_LENGTH + 0.62) * r_addr)
        elif len(edge_ports) == 2:
            pair = (min(edge_ports), max(edge_ports))
            pairs[pair] = pairs.get(pair, 0) + 1
            if pair[0] == pair[1]:
                reach = max(reach, (STUB_LENGTH + 0.62) * r_addr)
                fanned = True
        if len(edge_ports) >= 3 and len(set(edge_ports)) < len(edge_ports):
            fanned = True
    if fanned or any(count > 1 for count in pairs.values()):
        reach = max(reach, FAN_HEIGHT)
    return reach


def stub_direction(class_index: int, i: int) -> np.ndarray:
    """Deterministic unit direction (in the xy-plane) of an order-1 stub, so several stubs stay visible."""
    angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _order1_geom(anchor: np.ndarray, class_index: int, i: int, r_addr: float) -> ObjGeom:
    """A short stub leaving the address; its marker sits clear of the address circle."""
    tip = anchor + STUB_LENGTH * r_addr * stub_direction(class_index, i)
    return ObjGeom([np.stack([anchor, tip])], tip, [(anchor + tip) / 2.0])


def loop_direction(rank: tuple[int, int]) -> np.ndarray:
    """Unit direction (in the xy-plane) from an address to its ``rank``-th self-loop."""
    j, m = rank
    angle = 2.0 * np.pi * j / m + 0.6
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _loop_geom(anchor: np.ndarray, rank: tuple[int, int], r_addr: float) -> ObjGeom:
    """A self-loop drawn like a degenerate hub: a marker offset from the address and one fanned spoke per port.

    Both spokes start at the address center, so the loop stays attached whatever the zoom; several
    loops on one address spread around it.
    """
    marker = anchor + STUB_LENGTH * r_addr * loop_direction(rank)
    curves = [_fanned_curve(anchor, marker, -0.5), _fanned_curve(anchor, marker, 0.5)]
    return ObjGeom(curves, marker, [curve[8] for curve in curves])


def _fanned_curve(a: np.ndarray, b: np.ndarray, fan: float) -> np.ndarray:
    """A Bezier curve from ``a`` to ``b``, bent according to its rank among parallel connections."""
    chord = b - a
    length = max(float(np.linalg.norm(chord)), 1e-9)
    normal = _perpendicular(chord / length)
    height = fan * min(0.3 * length, FAN_HEIGHT)
    return _bezier(a, (a + b) / 2.0 + 2.0 * height * normal, b)


def _pair_geom(a: np.ndarray, b: np.ndarray, fan: float) -> ObjGeom:
    """A curve between two addresses, with the marker at its middle and one label per end."""
    curve = _fanned_curve(a, b, fan)
    return ObjGeom([curve], curve[8], [curve[3], curve[13]])


def _hub_geom(hub: np.ndarray, pos: np.ndarray, edge_ports: list[int]) -> ObjGeom:
    """A hub marker with one spoke per port; spokes to the same address are fanned out."""
    counts = {p: edge_ports.count(p) for p in edge_ports}
    seen: dict[int, int] = {}
    lines, labels = [], []
    for p in edge_ports:
        j = seen.get(p, 0)
        seen[p] = j + 1
        if counts[p] == 1:
            lines.append(np.stack([hub, pos[p]]))
            labels.append((hub + pos[p]) / 2.0)
        else:
            curve = _fanned_curve(hub, pos[p], j - (counts[p] - 1) / 2.0)
            lines.append(curve)
            labels.append(curve[8])
    return ObjGeom(lines, hub, labels)


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
                geoms[key] = _hub_geom(pos[data.hub_ids[key]], pos, edge_ports)
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
                if len(set(edge_ports)) == 1:  # all ports on one address: the hub is offset like a stub
                    item |= {"direction": stub_direction(class_index, i)[:2].round(6).tolist()}
            else:
                item |= {"kind": "none"}
            items.append(item)
        out[name] = items
    return out
