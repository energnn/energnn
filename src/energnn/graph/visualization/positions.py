# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 2 of the drawing pipeline: place the addresses and the hubs.

**Coordinates.** Everything is placed in *layout units*: a square box going from -1 to 1 on both axes,
whatever the size of the graph. The renderers map that box to pixels or to matplotlib data units. The
unit of every other size (marker radius, stub length, ...) is the *address radius*
(:func:`address_radius`), which shrinks with the number of addresses so that big graphs stay readable.

**Where do the coordinates come from?** In order of precedence:

1. ``address_positions``, one row per address, given by the user (latent coordinates, geography, ...);
2. the objects placed by ``hyper_edge_positions``: each address is put at the mean position of the placed
   objects pointing to it (a bus placed at its geographic coordinates places its own address);
3. otherwise a *spring layout* (:func:`spring_layout`): a classic physics simulation where every node
   repels every other one and connected nodes attract each other, which untangles the graph.

**Hubs.** Objects of order 3 or more are drawn as a marker (the *hub*) with one spoke to each address. The
hub sits at the barycenter of its distinct addresses, or, when the spring layout is used, wherever the
simulation put it (it takes part in it as one more node). A hub whose ports all hit the same address would
sit on that address and hide it, so it is offset like a stub.

The result is a :class:`Layout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energnn.graph.visualization.content import ObjKey, Topology, per_address_array

# --- sizes, in layout units (the box is [-1, 1]) or in address radii ------------------------------------------
#: Distance from an address to the marker of its stubs and loops, in address radii.
STUB_LENGTH = 2.6
#: Largest sideways bulge of a curve fanned out among parallel ones, in layout units.
FAN_HEIGHT = 0.09
#: Radius of a class marker, as a fraction of the address radius.
MARKER_RATIO = 0.62
#: A derived address closer than this many address radii to the object that placed it is pushed away.
HUB_CLEARANCE = 1.5
# The address radius is first computed in pixels of a reference 580 px canvas (half box: 290 px):
# 150 / sqrt(n) clipped to [5, 13] px, so that 4 addresses get 13 px and 900 get 5 px; then converted to
# layout units by dividing by the half box.
_RADIUS_PIXELS = (150.0, 5.0, 13.0, 290.0)


def address_radius(n_addresses: int) -> float:
    """Radius of the address circles in layout units, shrinking with the number of addresses.

    Every other size derives from it, so that stubs, markers and curves keep clear of the circles at any
    graph size.
    """
    scale, low, high, half_box = _RADIUS_PIXELS
    return float(np.clip(scale / np.sqrt(max(n_addresses, 1)), low, high)) / half_box


def stub_direction(class_index: int, i: int) -> np.ndarray:
    """Unit direction in which the stub of object ``i`` of class ``class_index`` leaves its address.

    Deterministic and spread over the circle (the fractional parts of ``0.37 * class_index + 0.61 * i``), so
    that several stubs on the same address, from the same or different classes, point in different
    directions and stay visible.
    """
    angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
    return np.array([np.cos(angle), np.sin(angle)])


@dataclass(frozen=True)
class Layout:
    """Where things are, in layout units.

    :param addresses: The address positions, shape ``(n_addresses, 2)``, inside the ``[-1, 1]`` box.
    :param hubs: The marker position of every object drawn as a hub (kinds ``"hub"`` and ``"placed"``), by
        object key, each of shape ``(2,)``.
    :param margin: How far beyond the box the drawings may reach (stubs, loops, fanned curves and markers
        stick out of the addresses), so that the renderers can enlarge their view accordingly.
    """

    addresses: np.ndarray
    hubs: dict[ObjKey, np.ndarray]
    margin: float


def spring_layout(n_nodes: int, edges: np.ndarray, *, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """
    Compute a Fruchterman-Reingold force-directed layout.

    The nodes start at random positions. At each iteration every node is pushed away from every other one
    (a repulsion decreasing with the distance) and pulled toward its neighbors (an attraction growing with
    the distance), and moves along the resulting force by at most a *temperature* that cools down to zero:
    large moves first to untangle, small ones at the end to settle. The result is centered and scaled into
    the ``[-1, 1]`` box.

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
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    if len(edges):
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]  # (n, n, 2): vector from every node to every other one
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, 1.0)  # a node exerts no force on itself
        dist = np.maximum(dist, 0.01)  # avoid infinite repulsion between nodes that landed on each other
        force = k * k / dist**2 - adjacency * dist / k  # repulsion for all pairs, attraction for neighbors
        displacement = (delta * force[..., None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-9)
        pos += displacement / length * np.minimum(length, temperature)  # move along the force, capped
        temperature -= cooling
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    return pos / scale if scale > 0 else pos


def lay_out(topology: Topology, *, address_positions: Any = None, iterations: int = 150, seed: int = 0) -> Layout:
    """Place the addresses and the hubs of ``topology`` (see the module docstring for the sources of coordinates).

    :param topology: The objects to place; the ones carrying a position are the "placed" objects.
    :param address_positions: Optional user coordinates, shape ``(n_addresses, 2)`` (padded length accepted).
    :param iterations: Spring layout iterations, when it is used.
    :param seed: Spring layout seed, when it is used.
    :return: The layout.
    :raises ValueError: If ``address_positions`` has a wrong shape or NaN rows, or if the placed objects leave
        an address without position.
    """
    placed = {h.key: h.position for h in topology.hyper_edges if h.position is not None}
    if address_positions is None and not placed:
        return _spring(topology, iterations, seed)

    # From here on the coordinates are the user's, in their own units: they are fitted into the box below.
    n = topology.n_addresses
    derived = np.zeros(n, dtype=bool)  # which addresses were placed by the objects (vs. given by the user)
    if address_positions is not None:
        addresses = per_address_array(address_positions, "address_positions", topology, (2,))
        if np.isnan(addresses).any():
            raise ValueError("address_positions holds NaN rows; every real address needs a position.")
    else:
        addresses, derived = _addresses_from_placed(topology, placed)

    # Fit everything that has user coordinates (addresses and placed objects) into the box together, with one
    # translation and one scale, so that the drawing keeps the user's geometry.
    fitted = np.concatenate([addresses, *[p[None] for p in placed.values()]])
    center = fitted.mean(axis=0)
    scale = float(np.abs(fitted - center).max()) or 1.0
    addresses = (addresses - center) / scale
    hubs = {key: (p - center) / scale for key, p in placed.items()}

    # A derived address sits exactly on the object that placed it (a bus places its own address), which would
    # hide the object's marker under the address circle: push the address away, like a stub.
    r_addr = address_radius(n)
    for h in topology.hyper_edges:
        if h.kind == "placed":
            for a in {p for p in h.ports if derived[p]}:
                if np.linalg.norm(addresses[a] - hubs[h.key]) < HUB_CLEARANCE * r_addr:
                    addresses[a] = hubs[h.key] + STUB_LENGTH * r_addr * stub_direction(0, a)
    _add_hubs(topology, addresses, hubs)
    return Layout(addresses, hubs, layout_margin(topology))


def _spring(topology: Topology, iterations: int, seed: int) -> Layout:
    """Lay out the graph with :func:`spring_layout` when no coordinate is given.

    The simulated graph has one node per address plus one node per hub (order-3+ object), each hub node
    being linked to the distinct addresses of its object; a pair links its two addresses directly. Stubs
    and loops involve a single address and play no role in the simulation.
    """
    n = topology.n_addresses
    edges: list[tuple[int, int]] = []
    hub_rows: dict[ObjKey, int] = {}  # object key -> its node index, after the n address nodes
    for h in topology.hyper_edges:
        if h.kind == "pair":
            edges.append((h.ports[0], h.ports[1]))
        elif h.kind == "hub":
            hub_rows[h.key] = n + len(hub_rows)
            edges.extend((hub_rows[h.key], p) for p in sorted(set(h.ports)))
    pos = spring_layout(n + len(hub_rows), np.array(edges, dtype=int).reshape(-1, 2), iterations=iterations, seed=seed)
    hubs = {key: pos[row] for key, row in hub_rows.items()}
    _add_hubs(topology, pos[:n], hubs)  # the hubs are already placed; only the single-address ones move
    return Layout(pos[:n], hubs, layout_margin(topology))


def _addresses_from_placed(topology: Topology, placed: dict[ObjKey, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Put each address at the mean position of the placed objects pointing to it.

    :return: The addresses, shape ``(n_addresses, 2)``, and the mask of derived addresses (all True here).
    :raises ValueError: If some address is pointed to by no placed object: it would have no position at all.
    """
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
    """Complete ``hubs`` in place with the hub of every order-3+ object.

    A hub already in ``hubs`` (placed by the spring layout) is kept, otherwise it goes to the barycenter of
    its distinct addresses. A hub whose ports all hit the same address is always offset from that address
    like a stub, since its barycenter would be the address itself.
    """
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
    """How far the drawings may reach beyond the ``[-1, 1]`` box, in layout units.

    The addresses are inside the box, but a stub or a loop sticks out of its address by its length plus
    its marker, a marker centered on an address sticks out by its radius, and fanned curves bulge sideways.
    The renderers enlarge their view by this margin so that nothing is cut.
    """
    r_addr = address_radius(topology.n_addresses)
    reach = MARKER_RATIO * r_addr  # at least a class marker centered on an address
    pairs: dict[tuple[int, int], int] = {}  # how many pairs share each address pair, to detect fanning
    for h in topology.hyper_edges:
        if h.kind in ("stub", "loop") or (h.kind == "hub" and len(set(h.ports)) == 1):
            reach = max(reach, (STUB_LENGTH + MARKER_RATIO) * r_addr)  # stub-like: length plus the marker
        if h.kind == "pair":
            pairs[(min(h.ports), max(h.ports))] = pairs.get((min(h.ports), max(h.ports)), 0) + 1
        if h.kind in ("loop", "hub") and len(set(h.ports)) < len(h.ports):
            reach = max(reach, FAN_HEIGHT)  # repeated ports: fanned spokes
    if any(count > 1 for count in pairs.values()):
        reach = max(reach, FAN_HEIGHT)  # parallel pairs: fanned curves
    return reach
