# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

HYPER_EDGE_SETS = "hyper_edge_sets"
FEATURE_LIST = "feature_list"
PORT_LIST = "port_list"


class HyperEdgeSetStructure(dict):
    """Edge structure specification."""

    def __init__(self, *, port_list: list[str] | None, feature_list: list[str] | None):
        super().__init__()
        self[PORT_LIST] = port_list
        self[FEATURE_LIST] = feature_list

    @classmethod
    def from_list(cls, *, port_list: list[str] | None, feature_list: list[str] | None) -> "HyperEdgeSetStructure":
        return cls(port_list=port_list, feature_list=feature_list)

    @property
    def port_list(self) -> list[str] | None:
        return self[PORT_LIST]

    @property
    def feature_list(self) -> list[str] | None:
        return self[FEATURE_LIST]


class GraphStructure(dict):
    """Graph structure specification."""

    def __init__(self, hyper_edge_sets: dict[str, HyperEdgeSetStructure]):
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets

    @classmethod
    def from_dict(cls, *, hyper_edge_set_structure_dict: dict[str, HyperEdgeSetStructure]) -> "GraphStructure":
        return cls(hyper_edge_set_structure_dict)

    @property
    def hyper_edge_sets(self) -> dict[str, HyperEdgeSetStructure]:
        return self[HYPER_EDGE_SETS]

    def __str__(self):
        items = list(self.hyper_edge_sets.items())
        lines = [f"GraphStructure · {len(items)} hyper-edge set{'s' if len(items) != 1 else ''}"]
        if not items:
            return lines[0]

        name_width = max(len(name) for name, _ in items)
        port_cells = ["ports: " + (", ".join(s.port_list) if s.port_list else "—") for _, s in items]
        port_width = max(len(cell) for cell in port_cells)
        for i, ((name, structure), ports) in enumerate(zip(items, port_cells)):
            branch = "└─ " if i == len(items) - 1 else "├─ "
            features = "features: " + (", ".join(structure.feature_list) if structure.feature_list else "—")
            lines.append(f"{branch}{name.ljust(name_width)}   {ports.ljust(port_width)}   {features}")
        return "\n".join(lines)
