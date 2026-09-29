# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 2 of the pipeline: place the addresses and the hubs in the ``[-1, 1]`` box.

Three sources of coordinates, in this order of precedence: ``address_positions`` (one row per address),
the objects placed by ``hyper_edge_positions`` (each address at the mean position of the placed objects
pointing to it), and otherwise a spring layout of the whole graph. Hubs of order-3+ objects sit at the
barycenter of their distinct ports. All the sizes below are in layout units, where the box is ``[-1, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energnn.graph.visualization.content import ObjKey, Topology, per_address_array

# --- sizes, in layout units -------------------------------------------------------------------
STUB_LENGTH = 2.6  # distance from an address to the marker of its stubs and loops, in address radii
FAN_HEIGHT = 0.09  # largest bulge of a fanned-out parallel curve
MARKER_RATIO = 0.62  # class marker radius, as a fraction of the address radius
HUB_CLEARANCE = 1.5  # a hub closer than this many address radii to one of its addresses is pushed away
_RADIUS_PIXELS = (150.0, 5.0, 13.0, 290.0)  # address radius in pixels: 150 / sqrt(n) clipped to [5, 13], on a 290 px half box


def address_radius(n_addresses: int) -> float:
    """Radius of the address circles, shrinking with their number, so that geometries can keep clear of them."""
    scale, low, high, half_box = _RADIUS_PIXELS
    return float(np.clip(scale / np.sqrt(max(n_addresses, 1)), low, high)) / half_box


def stub_direction(class_index: int, i: int) -> np.ndarray:
    """Deterministic unit direction of an order-1 stub (or of an offset hub), so that several stay visible."""
    angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
    return np.array([np.cos(angle), np.sin(angle)])


@dataclass(frozen=True)
class Layout:
    """Where things are, in the ``[-1, 1]`` box."""

    addresses: np.ndarray  # (n_addresses, 2)
    hubs: dict[ObjKey, np.ndarray]  # marker anchor of every hub and placed object, shape (2,)
    margin: float  # how far stubs, loops, fans and markers may reach beyond the box


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


def lay_out(topology: Topology, *, address_positions: Any = None, iterations: int = 150, seed: int = 0) -> Layout:
    """Place the addresses and the hubs of ``topology``.

    :raises ValueError: If ``address_positions`` has a wrong shape or NaN rows, or if an address gets no position
        from the placed objects.
    """
    placed = {h.key: h.position for h in topology.hyper_edges if h.position is not None}
    if address_positions is None and not placed:
        return _spring(topology, iterations, seed)
    n = topology.n_addresses
    derived = np.zeros(n, dtype=bool)
    if address_positions is not None:
        addresses = per_address_array(address_positions, "address_positions", topology, (2,))
        if np.isnan(addresses).any():
            raise ValueError("address_positions holds NaN rows; every real address needs a position.")
    else:
        addresses, derived = _addresses_from_placed(topology, placed)
    # fit the addresses and the placed objects together in the box
    fitted = np.concatenate([addresses, *[p[None] for p in placed.values()]])
    center = fitted.mean(axis=0)
    scale = float(np.abs(fitted - center).max()) or 1.0
    addresses = (addresses - center) / scale
    hubs = {key: (p - center) / scale for key, p in placed.items()}
    r_addr = address_radius(n)
    for h in topology.hyper_edges:  # a derived address sitting on a placed object would hide it: push it away
        if h.kind == "placed":
            for a in {p for p in h.ports if derived[p]}:
                if np.linalg.norm(addresses[a] - hubs[h.key]) < HUB_CLEARANCE * r_addr:
                    addresses[a] = hubs[h.key] + STUB_LENGTH * r_addr * stub_direction(0, a)
    _add_hubs(topology, addresses, hubs)
    return Layout(addresses, hubs, layout_margin(topology))


def _spring(topology: Topology, iterations: int, seed: int) -> Layout:
    """Spring layout over the addresses plus one node per hub, linked to its distinct ports."""
    n = topology.n_addresses
    edges: list[tuple[int, int]] = []
    hub_rows: dict[ObjKey, int] = {}
    for h in topology.hyper_edges:
        if h.kind == "pair":
            edges.append((h.ports[0], h.ports[1]))
        elif h.kind == "hub":
            hub_rows[h.key] = n + len(hub_rows)
            edges.extend((hub_rows[h.key], p) for p in sorted(set(h.ports)))
    pos = spring_layout(n + len(hub_rows), np.array(edges, dtype=int).reshape(-1, 2), iterations=iterations, seed=seed)
    hubs = {key: pos[row] for key, row in hub_rows.items()}
    _add_hubs(topology, pos[:n], hubs)  # only the hubs with a single distinct address move
    return Layout(pos[:n], hubs, layout_margin(topology))


def _addresses_from_placed(topology: Topology, placed: dict[ObjKey, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Each address at the mean position of the placed objects pointing to it."""
    n = topology.n_addresses
    addresses, hits = np.zeros((n, 2)), np.zeros(n)
    for h in topology.hyper_edges:
        if h.kind == "placed":
            for p in h.ports:
                addresses[p] += placed[h.key]
                hits[p] += 1
    if not (hits > 0).all():
        unplaced = np.flatnonzero(hits == 0).tolist()
        raise ValueError(
            f"addresses {unplaced} are pointed to by no placed object, so they have no position; "
            "give address_positions, or place a class whose ports cover every address."
        )
    return addresses / hits[:, None], np.ones(n, dtype=bool)


def _add_hubs(topology: Topology, addresses: np.ndarray, hubs: dict[ObjKey, np.ndarray]) -> None:
    """Hubs of order-3+ objects at the barycenter of their distinct ports (kept where the spring layout put them
    when already known); a hub whose ports all hit one address is offset from it like a stub."""
    r_addr = address_radius(topology.n_addresses)
    for class_index, cls in enumerate(topology.classes):
        for h in topology.of(cls):
            if h.kind != "hub":
                continue
            distinct = sorted(set(h.ports))
            if len(distinct) == 1:
                hubs[h.key] = addresses[distinct[0]] + STUB_LENGTH * r_addr * stub_direction(class_index, h.index)
            elif h.key not in hubs:
                hubs[h.key] = addresses[distinct].mean(axis=0)


def layout_margin(topology: Topology) -> float:
    """Room to keep around the box so that stubs, self-loops, fanned curves and markers stay in view."""
    r_addr = address_radius(topology.n_addresses)
    reach = MARKER_RATIO * r_addr  # a class marker centered on an address
    pairs: dict[tuple[int, int], int] = {}
    for h in topology.hyper_edges:
        if h.kind in ("stub", "loop") or (h.kind == "hub" and len(set(h.ports)) == 1):
            reach = max(reach, (STUB_LENGTH + MARKER_RATIO) * r_addr)
        if h.kind == "pair":
            pairs[(min(h.ports), max(h.ports))] = pairs.get((min(h.ports), max(h.ports)), 0) + 1
        if h.kind in ("loop", "hub") and len(set(h.ports)) < len(h.ports):
            reach = max(reach, FAN_HEIGHT)
    if any(count > 1 for count in pairs.values()):
        reach = max(reach, FAN_HEIGHT)
    return reach
