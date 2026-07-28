# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest

from energnn.converter import ElementsConverter


class StubElementsConverter(ElementsConverter):
    """Returns a fixed table, regardless of the input."""

    def __init__(self, table: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.table = table

    def _get_table(self, *args, **kwargs) -> pd.DataFrame:
        return self.table


@pytest.fixture
def table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "from": ["bus_a", "bus_b"],
            "to": ["bus_b", "bus_c"],
            "susceptance": [1.5, 2.5],
            "status": ["open", "closed"],
        }
    )


def test_both_none_raises(table):
    with pytest.raises(ValueError):
        StubElementsConverter(table, port_list=None, feature_list=None)


def test_call_splits_ports_and_features(table):
    converter = StubElementsConverter(table, port_list=["from", "to"], feature_list=["susceptance", "status"])
    df_port, df_feature = converter()

    assert list(df_port.columns) == ["from", "to"]
    assert list(df_feature.columns) == ["susceptance", "status"]
    assert len(df_port) == len(df_feature) == 2
    assert df_port["from"].tolist() == ["bus_a", "bus_b"]
    assert df_feature["susceptance"].tolist() == [1.5, 2.5]


def test_call_ports_only(table):
    converter = StubElementsConverter(table, port_list=["from", "to"], feature_list=None)
    df_port, df_feature = converter()

    assert df_feature is None
    assert list(df_port.columns) == ["from", "to"]


def test_call_features_only(table):
    converter = StubElementsConverter(table, port_list=None, feature_list=["susceptance"])
    df_port, df_feature = converter()

    assert df_port is None
    assert list(df_feature.columns) == ["susceptance"]


def test_attributes_concatenates_ports_and_features(table):
    converter = StubElementsConverter(table, port_list=["from", "to"], feature_list=["susceptance"])
    assert converter.attributes == ["from", "to", "susceptance"]


def test_get_structure_round_trip(table):
    converter = StubElementsConverter(table, port_list=["from", "to"], feature_list=["susceptance"])
    structure = converter.get_structure()

    assert structure.port_list == ["from", "to"]
    assert structure.feature_list == ["susceptance"]

    structure_no_ports = StubElementsConverter(table, port_list=None, feature_list=["susceptance"]).get_structure()
    assert structure_no_ports.port_list is None
    assert structure_no_ports.feature_list == ["susceptance"]
