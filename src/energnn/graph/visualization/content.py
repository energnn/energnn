# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 1 of the drawing pipeline: read the real (non-fictitious) objects of a single Graph.

**Why this step exists.** A :class:`~energnn.graph.Graph` stores its data as arrays, padded with
fictitious objects and addresses so that graphs of different sizes can be batched. Drawing needs the
opposite: one record per real object, with its ports and features spelled out. This module does that
conversion once, and nothing downstream touches the Graph again. The result is a :class:`Topology`
made of :class:`HyperEdge` records.

**Vocabulary** (see also the Basics page of the documentation):

- an *address* is an integer node of the graph; addresses carry no data, they only connect objects;
- a *hyper-edge* (or *object*) belongs to a *class* (``"bus"``, ``"line"``, ...), points to zero, one or
  more addresses through its *ports* (``"from"``, ``"to"``, ...) and carries numerical *features*;
- the *order* of an object is its number of ports.

**How an object is drawn** is decided here too, once and for all, by :attr:`HyperEdge.kind`: every later
step (placement, geometry, rendering) dispatches on that kind instead of re-deriving it from the ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

#: Identifies one hyper-edge: ``(class name, index among the real objects of that class)``.
ObjKey = tuple[str, int]
#: ``{class: [feature names]}``, the form of ``hyper_edge_positions`` and ``hyper_edge_colors``: which
#: features of which class hold the coordinates or the color channels of its objects.
FeatureSpec = dict[str, list[str]]
#: The drawing rule of an object, see :attr:`HyperEdge.kind`.
Kind = Literal["none", "stub", "pair", "loop", "hub", "placed"]


@dataclass(frozen=True)
class HyperEdge:
    """One real object of the graph, with everything the drawing needs to know about it.

    :param cls: Name of its hyper-edge class.
    :param index: Its index among the real objects of that class (fictitious objects are not counted).
    :param ports: The addresses it points to, in the sorted order of the port names of its class.
    :param features: Its feature values, by feature name.
    :param position: The raw coordinates read from its features when ``hyper_edge_positions`` names its
        class, otherwise None. Raw means "as found in the graph": the fit into the drawing box is done later.
    """

    cls: str
    index: int
    ports: tuple[int, ...]
    features: dict[str, float]
    position: np.ndarray | None = None

    @property
    def key(self) -> ObjKey:
        """The ``(class, index)`` pair that identifies this object in every dictionary of the pipeline."""
        return (self.cls, self.index)

    @property
    def kind(self) -> Kind:
        """How this object is drawn.

        - ``"placed"``: it carries a position, so it is drawn as a marker at that position with one line
          (a *spoke*) to each of its addresses, whatever its order;
        - ``"none"``: no port, nothing to draw (a decision graph's objects, for instance);
        - ``"stub"``: one port, a short segment leaving its address with a marker at the tip;
        - ``"pair"``: two distinct ports, a curve between the two addresses with a marker in the middle;
        - ``"loop"``: two ports on the same address (a self-loop), a marker beside the address with two spokes;
        - ``"hub"``: three ports or more, a marker at the barycenter of its addresses with one spoke each.
        """
        if self.position is not None:
            return "placed"
        if len(self.ports) == 0:
            return "none"
        if len(self.ports) == 1:
            return "stub"
        if len(self.ports) == 2:
            return "loop" if self.ports[0] == self.ports[1] else "pair"
        return "hub"


@dataclass(frozen=True)
class Topology:
    """The real content of a single Graph, as read by :func:`read_graph`.

    :param address_mask: Boolean array over the graph's (possibly padded) address registry: True for the real
        addresses. Its length is the padded count, its sum the real one; it lets per-address user arrays be
        given at either length.
    :param classes: The class names, sorted, which fixes the class order used everywhere (colors, markers).
    :param port_names: For each class, its port names, sorted; :attr:`HyperEdge.ports` follows that order.
    :param hyper_edges: Every real object, classes in the order of ``classes``, objects in index order.
    """

    address_mask: np.ndarray
    classes: list[str]
    port_names: dict[str, list[str]]
    hyper_edges: list[HyperEdge]

    @property
    def n_addresses(self) -> int:
        """Number of real addresses."""
        return int(self.address_mask.sum())

    def of(self, cls: str) -> list[HyperEdge]:
        """The objects of one class, in index order."""
        return [h for h in self.hyper_edges if h.cls == cls]


def read_graph(graph: Graph, *, hyper_edge_positions: FeatureSpec | None = None) -> Topology:
    """Read a single Graph into a :class:`Topology`.

    Fictitious (padded) objects and addresses are dropped. When ``hyper_edge_positions`` names a class, its
    objects get their :attr:`~HyperEdge.position` from the two named features; an object with a NaN
    coordinate is left without position, hence drawn like any other object of its order.

    :param graph: A single (non-batched) Graph, on any backend.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}``.
    :return: The topology.
    :raises ValueError: If the graph is batched, or if ``hyper_edge_positions`` names an unknown class or
        feature or does not list exactly two features.
    """
    if not graph.is_single:
        raise ValueError("Only single graphs can be drawn; use separate_graphs() on a batch first.")
    g = graph.to_numpy_backend()  # the pipeline is numpy-only, whatever backend the graph lives on
    classes = sorted(g.hyper_edge_sets)
    port_names: dict[str, list[str]] = {}
    hyper_edges: list[HyperEdge] = []
    for cls in classes:
        hes = g.hyper_edge_sets[cls]
        real = np.asarray(hes.non_fictitious) > 0  # one flag per object row, False for the padding
        port_names[cls] = sorted(hes.port_dict) if hes.port_dict is not None else []
        port_dict, feature_dict = hes.port_dict or {}, hes.feature_names or {}
        # ports: one column per port name, real rows only -> (n_real, n_ports); None for a port-less class
        ports = np.stack([np.asarray(port_dict[k])[real] for k in port_names[cls]], axis=-1) if port_names[cls] else None
        # features are stored as one array with one column per feature; feature_names maps a name to its column
        feature_names = sorted(feature_dict.items())
        feature_array = np.asarray(hes.feature_array)[real] if feature_names else np.zeros((int(real.sum()), 0))
        for i in range(int(real.sum())):
            hyper_edges.append(
                HyperEdge(
                    cls=cls,
                    index=i,
                    ports=tuple(int(p) for p in ports[i]) if ports is not None else (),
                    features={name: float(feature_array[i, int(idx)]) for name, idx in feature_names},
                )
            )
    topology = Topology(np.asarray(g.non_fictitious_addresses) > 0, classes, port_names, hyper_edges)
    if hyper_edge_positions:
        # second pass: attach the coordinates; the records are frozen, so placed objects are rebuilt
        columns = feature_columns(topology, hyper_edge_positions, "hyper_edge_positions", (2,))
        placed = []
        for h in topology.hyper_edges:
            coords = columns[h.cls][h.index] if h.cls in columns else None
            if coords is not None and np.isfinite(coords).all():
                h = HyperEdge(h.cls, h.index, h.ports, h.features, coords)
            placed.append(h)
        topology = Topology(topology.address_mask, classes, port_names, placed)
    return topology


def feature_columns(topology: Topology, spec: FeatureSpec, what: str, widths: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Read the features named by a ``{class: [feature names]}`` spec into one array per class.

    Used for both ``hyper_edge_positions`` (two names: x and y) and ``hyper_edge_colors`` (one or two names:
    the color channels).

    :param topology: Where the objects and their features come from.
    :param spec: ``{class: [feature names]}``.
    :param what: The parameter name, for error messages.
    :param widths: The allowed numbers of names per class.
    :return: ``{class: array of shape (n_objects of that class, number of names)}``. NaN values pass through.
    :raises ValueError: On an unknown class or feature, a wrong number of names, or classes listing different
        numbers of names (every class must give the same number of coordinates or channels).
    """
    columns: dict[str, np.ndarray] = {}
    for cls, names in spec.items():
        if cls not in topology.classes:
            raise ValueError(f"{what}: unknown hyper-edge class '{cls}'; the graph has {topology.classes}.")
        names = list(names)
        if len(names) not in widths:
            raise ValueError(f"{what}['{cls}'] must list {' or '.join(map(str, widths))} feature names; got {names}.")
        objects = topology.of(cls)
        known = sorted(objects[0].features) if objects else []  # every object of a class has the same features
        unknown = [name for name in names if objects and name not in objects[0].features]
        if unknown:
            raise ValueError(f"{what}['{cls}']: class '{cls}' has no feature {unknown}; its features are {known}.")
        columns[cls] = np.array([[h.features[name] for name in names] for h in objects], dtype=float).reshape(-1, len(names))
    if len({c.shape[1] for c in columns.values()}) > 1:
        raise ValueError(f"{what}: every class must list the same number of feature names; got {spec}.")
    return columns


def per_address_array(values: Any, name: str, topology: Topology, widths: tuple[int, ...]) -> np.ndarray:
    """Validate a per-address user array (``address_positions`` or ``address_colors``) and drop its fictitious rows.

    :param values: Anything ``np.asarray`` accepts, of shape ``(n, k)``.
    :param name: The parameter name, for error messages.
    :param topology: Gives the real and padded address counts.
    :param widths: The allowed values of ``k`` (2 for coordinates, 1 or 2 for color channels).
    :return: A float array of shape ``(n_addresses, k)`` over the real addresses. The user may give one row per
        real address, or one row per address of the padded registry (then the fictitious rows are dropped).
    :raises ValueError: On a wrong number of dimensions, columns or rows.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] not in widths:
        raise ValueError(f"{name} must have shape (n_addresses, k) with k in {widths}; got {array.shape}.")
    n_current, n_real = len(topology.address_mask), topology.n_addresses
    if array.shape[0] == n_current:
        array = array[topology.address_mask]
    if array.shape[0] != n_real:
        raise ValueError(f"{name} must have {n_real} (real) or {n_current} (current) rows; got {array.shape[0]}.")
    return array
