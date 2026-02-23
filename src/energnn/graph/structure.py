# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

EDGES = "edges"
FEATURE_LIST = "feature_list"
ADDRESS_LIST = "address_list"


class EdgeStructure(dict):
    """Edge structure specification."""

    def __init__(self, *, address_list: list[str] | None, feature_list: list[str] | None):
        super().__init__()
        self[ADDRESS_LIST] = address_list
        self[FEATURE_LIST] = feature_list

    @classmethod
    def from_list(cls, *, address_list: list[str] | None, feature_list: list[str] | None) -> "EdgeStructure":
        return cls(address_list=address_list, feature_list=feature_list)

    @property
    def address_list(self) -> list[str] | None:
        return self[ADDRESS_LIST]

    @property
    def feature_list(self) -> list[str] | None:
        return self[FEATURE_LIST]


class GraphStructure(dict):
    """Graph structure specification."""

    def __init__(self, edges: dict[str, EdgeStructure]):
        super().__init__()
        self[EDGES] = edges

    @classmethod
    def from_dict(cls, *, edge_structure_dict: dict[str, EdgeStructure]) -> "GraphStructure":
        return cls(edge_structure_dict)

    @property
    def edges(self) -> dict[str, EdgeStructure]:
        return self[EDGES]

    def __str__(self):
        data = {
            "Name": [edge_name for edge_name in self.edges.keys()],
            "Addresses": [edge_structure.address_list for edge_structure in self.edges.values()],
            "Features": [edge_structure.feature_list for edge_structure in self.edges.values()],
        }
        df = pd.DataFrame(data).set_index("Name")
        return df.to_string()
