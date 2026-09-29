# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 1 of the pipeline: read the real (non-fictitious) objects of a single Graph.

The result is a :class:`Topology`: the addresses, the classes with their port names, and one
:class:`HyperEdge` record per object holding its ports, its features and, when
``hyper_edge_positions`` names its class, the coordinates read from its features. Every later
step works on these records; nothing downstream touches the Graph again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

# ``(class name, index among the real objects of that class)``: identifies one hyper-edge.
ObjKey = tuple[str, int]
# ``{class: [feature names]}``: coordinates or color channels read from the features of a class.
FeatureSpec = dict[str, list[str]]
# How an object is drawn, decided once from its ports and whether it carries a position (see HyperEdge.kind).
Kind = Literal["none", "stub", "pair", "loop", "hub", "placed"]


@dataclass(frozen=True)
class HyperEdge:
    """One real object of the graph."""

    cls: str
    index: int
    ports: tuple[int, ...]  # port addresses, in the sorted order of the port names
    features: dict[str, float]  # feature name -> value
    position: np.ndarray | None = None  # raw coordinates read from ``hyper_edge_positions``, or None

    @property
    def key(self) -> ObjKey:
        return (self.cls, self.index)

    @property
    def kind(self) -> Kind:
        """The drawing rule: a placed object is a hub at its position whatever its order; otherwise the order
        decides (a stub attached to its address, a curve between two addresses, a self-loop, or a hub with spokes).
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
    """Real content of a single Graph."""

    address_mask: np.ndarray  # (n_current,) bool: which rows of the padded address registry are real
    classes: list[str]  # sorted class names
    port_names: dict[str, list[str]]  # class -> sorted port names
    hyper_edges: list[HyperEdge]  # every real object, classes in order, objects in index order

    @property
    def n_addresses(self) -> int:
        return int(self.address_mask.sum())

    def of(self, cls: str) -> list[HyperEdge]:
        return [h for h in self.hyper_edges if h.cls == cls]


def read_graph(graph: Graph, *, hyper_edge_positions: FeatureSpec | None = None) -> Topology:
    """Collect the real objects of a single Graph; objects of the classes listed in ``hyper_edge_positions``
    carry the coordinates read from the two named features (an object with a NaN coordinate is left unplaced).

    :raises ValueError: If the graph is batched, or if ``hyper_edge_positions`` names an unknown class or feature.
    """
    if not graph.is_single:
        raise ValueError("Only single graphs can be drawn; use separate_graphs() on a batch first.")
    g = graph.to_numpy_backend()
    classes = sorted(g.hyper_edge_sets)
    port_names: dict[str, list[str]] = {}
    hyper_edges: list[HyperEdge] = []
    for cls in classes:
        hes = g.hyper_edge_sets[cls]
        real = np.asarray(hes.non_fictitious) > 0
        port_names[cls] = sorted(hes.port_dict) if hes.port_dict is not None else []
        port_dict, feature_dict = hes.port_dict or {}, hes.feature_names or {}
        ports = np.stack([np.asarray(port_dict[k])[real] for k in port_names[cls]], axis=-1) if port_names[cls] else None
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
    """Read ``{class: [feature names]}`` -> ``{class: (n_objects, len(names))}`` from the objects' features.

    :raises ValueError: On an unknown class or feature, a wrong number of names, or inconsistent widths.
    """
    columns: dict[str, np.ndarray] = {}
    for cls, names in spec.items():
        if cls not in topology.classes:
            raise ValueError(f"{what}: unknown hyper-edge class '{cls}'; the graph has {topology.classes}.")
        names = list(names)
        if len(names) not in widths:
            raise ValueError(f"{what}['{cls}'] must list {' or '.join(map(str, widths))} feature names; got {names}.")
        objects = topology.of(cls)
        known = sorted(objects[0].features) if objects else []
        unknown = [name for name in names if objects and name not in objects[0].features]
        if unknown:
            raise ValueError(f"{what}['{cls}']: class '{cls}' has no feature {unknown}; its features are {known}.")
        columns[cls] = np.array([[h.features[name] for name in names] for h in objects], dtype=float).reshape(-1, len(names))
    if len({c.shape[1] for c in columns.values()}) > 1:
        raise ValueError(f"{what}: every class must list the same number of feature names; got {spec}.")
    return columns


def per_address_array(values: Any, name: str, topology: Topology, widths: tuple[int, ...]) -> np.ndarray:
    """Validate a per-address array of shape ``(n_addresses, k)`` or ``(n_current, k)`` and drop the fictitious rows."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] not in widths:
        raise ValueError(f"{name} must have shape (n_addresses, k) with k in {widths}; got {array.shape}.")
    n_current, n_real = len(topology.address_mask), topology.n_addresses
    if array.shape[0] == n_current:
        array = array[topology.address_mask]
    if array.shape[0] != n_real:
        raise ValueError(f"{name} must have {n_real} (real) or {n_current} (current) rows; got {array.shape[0]}.")
    return array
