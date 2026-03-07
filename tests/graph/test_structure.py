#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from energnn.graph import GraphStructure, HyperEdgeSetStructure


def test_edge_structure_init():
    port_list = ["from", "to"]
    feature_list = ["feat1", "feat2"]

    es = HyperEdgeSetStructure(port_list=port_list, feature_list=feature_list)
    assert es.port_list == port_list
    assert es.feature_list == feature_list
    assert es["port_list"] == port_list
    assert es["feature_list"] == feature_list

    es_none = HyperEdgeSetStructure(port_list=None, feature_list=None)
    assert es_none.port_list is None
    assert es_none.feature_list is None


def test_edge_structure_from_list():
    port_list = ["id"]
    feature_list = ["val"]
    es = HyperEdgeSetStructure.from_list(port_list=port_list, feature_list=feature_list)
    assert isinstance(es, HyperEdgeSetStructure)
    assert es.port_list == port_list
    assert es.feature_list == feature_list


def test_graph_structure_init():
    es1 = HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["val"])
    es2 = HyperEdgeSetStructure(port_list=["id"], feature_list=["state"])
    edges = {"arrow": es1, "node": es2}

    gs = GraphStructure(hyper_edge_sets=edges)
    assert gs.hyper_edge_sets == edges
    assert gs["hyper_edge_sets"] == edges
    assert gs.hyper_edge_sets["arrow"] is es1
    assert gs.hyper_edge_sets["node"] is es2


def test_graph_structure_from_dict():
    es = HyperEdgeSetStructure(port_list=["id"], feature_list=None)
    edge_dict = {"source": es}
    gs = GraphStructure.from_dict(hyper_edge_set_structure_dict=edge_dict)
    assert isinstance(gs, GraphStructure)
    assert gs.hyper_edge_sets == edge_dict
    assert gs.hyper_edge_sets["source"] is es


def test_structure_inheritance_dict():
    es = HyperEdgeSetStructure(port_list=[], feature_list=[])
    assert isinstance(es, dict)

    gs = GraphStructure(hyper_edge_sets={"e": es})
    assert isinstance(gs, dict)
    assert "hyper_edge_sets" in gs
