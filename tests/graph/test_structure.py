#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from energnn.graph import GraphStructure, EdgeStructure


def test_edge_structure_init():
    address_list = ["from", "to"]
    feature_list = ["feat1", "feat2"]

    es = EdgeStructure(address_list=address_list, feature_list=feature_list)
    assert es.address_list == address_list
    assert es.feature_list == feature_list
    assert es["address_list"] == address_list
    assert es["feature_list"] == feature_list

    es_none = EdgeStructure(address_list=None, feature_list=None)
    assert es_none.address_list is None
    assert es_none.feature_list is None


def test_edge_structure_from_list():
    address_list = ["id"]
    feature_list = ["val"]
    es = EdgeStructure.from_list(address_list=address_list, feature_list=feature_list)
    assert isinstance(es, EdgeStructure)
    assert es.address_list == address_list
    assert es.feature_list == feature_list


def test_graph_structure_init():
    es1 = EdgeStructure(address_list=["from", "to"], feature_list=["val"])
    es2 = EdgeStructure(address_list=["id"], feature_list=["state"])
    edges = {"arrow": es1, "node": es2}

    gs = GraphStructure(edges=edges)
    assert gs.edges == edges
    assert gs["edges"] == edges
    assert gs.edges["arrow"] is es1
    assert gs.edges["node"] is es2


def test_graph_structure_from_dict():
    es = EdgeStructure(address_list=["id"], feature_list=None)
    edge_dict = {"source": es}
    gs = GraphStructure.from_dict(edge_structure_dict=edge_dict)
    assert isinstance(gs, GraphStructure)
    assert gs.edges == edge_dict
    assert gs.edges["source"] is es


def test_structure_inheritance_dict():
    es = EdgeStructure(address_list=[], feature_list=[])
    assert isinstance(es, dict)

    gs = GraphStructure(edges={"e": es})
    assert isinstance(gs, dict)
    assert "edges" in gs
