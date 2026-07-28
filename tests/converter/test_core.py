# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from energnn.converter import Converter, ElementsConverter
from energnn.converter.core import _any_to_float, _str_to_int
from energnn.graph import GraphShape, collate_graphs
from energnn.graph.backend import JaxBackend, NumpyBackend


class _LineConverter(ElementsConverter):
    def _get_table(self, data):
        return data["line"]


class _BusConverter(ElementsConverter):
    def _get_table(self, data):
        return data["bus"]


class DummyConverter(Converter):
    __test__ = False

    def __init__(self, backend=None):
        self.backend = backend
        self.elements_converter_dict = {
            "line": _LineConverter(port_list=["from", "to"], feature_list=["susceptance", "status"]),
            "bus": _BusConverter(port_list=["id"], feature_list=["active_power"]),
        }


def make_data(n_bus: int = 3) -> dict:
    """A small network whose buses are shared between the line and bus tables."""
    bus_ids = [f"bus_{i}" for i in range(n_bus)]
    return {
        "line": pd.DataFrame(
            {
                "from": bus_ids[:-1],
                "to": bus_ids[1:],
                "susceptance": np.linspace(1.0, 2.0, n_bus - 1),
                "status": ["open", "closed"] * ((n_bus - 1) // 2) + ["open"] * ((n_bus - 1) % 2),
            }
        ),
        "bus": pd.DataFrame({"id": bus_ids, "active_power": np.linspace(-1.0, 1.0, n_bus)}),
    }


# ---------------------------------------------------------------------------
# End-to-end conversion
# ---------------------------------------------------------------------------


def test_address_mapping_consistency():
    """A given address must receive the same integer in every hyper-edge set."""
    graph = DummyConverter()(make_data())

    bus_ports = graph.hyper_edge_sets["bus"].port_dict["id"]
    line_from = graph.hyper_edge_sets["line"].port_dict["from"]
    line_to = graph.hyper_edge_sets["line"].port_dict["to"]

    # bus_i rows are in table order, so bus_ports[i] is the integer assigned to "bus_i".
    address_of = {f"bus_{i}": bus_ports[i] for i in range(3)}
    assert line_from.tolist() == [address_of["bus_0"], address_of["bus_1"]]
    assert line_to.tolist() == [address_of["bus_1"], address_of["bus_2"]]


def test_addresses_cover_consecutive_integers():
    graph = DummyConverter()(make_data(n_bus=5))

    all_ports = np.concatenate(
        [np.asarray(v) for hes in graph.hyper_edge_sets.values() if hes.port_dict for v in hes.port_dict.values()]
    )
    assert set(all_ports.astype(int).tolist()) == set(range(5))
    assert int(graph.true_shape.addresses) == 5


def test_determinism_across_instances():
    data = make_data()
    graph_1 = DummyConverter()(data)
    graph_2 = DummyConverter()(data)

    for key in graph_1.hyper_edge_sets:
        hes_1, hes_2 = graph_1.hyper_edge_sets[key], graph_2.hyper_edge_sets[key]
        if hes_1.port_dict is not None:
            for port in hes_1.port_dict:
                assert np.array_equal(hes_1.port_dict[port], hes_2.port_dict[port])
        assert np.array_equal(hes_1.feature_array, hes_2.feature_array)
        assert hes_1.feature_names == hes_2.feature_names


def test_get_structure():
    structure = DummyConverter().get_structure()

    assert set(structure.hyper_edge_sets.keys()) == {"line", "bus"}
    assert structure.hyper_edge_sets["line"].port_list == ["from", "to"]
    assert structure.hyper_edge_sets["line"].feature_list == ["susceptance", "status"]
    assert structure.hyper_edge_sets["bus"].port_list == ["id"]


def test_input_not_mutated():
    data = make_data()
    original_status = data["line"]["status"].copy()
    DummyConverter()(data)

    assert data["line"]["status"].dtype == object
    assert data["line"]["status"].equals(original_status)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def test_default_backend_is_numpy():
    graph = DummyConverter()(make_data())
    assert graph._backend == NumpyBackend()
    assert isinstance(graph.hyper_edge_sets["bus"].feature_array, np.ndarray)


@pytest.mark.parametrize("backend", [NumpyBackend(), JaxBackend()], ids=["numpy", "jax"])
def test_target_backend(backend):
    graph = DummyConverter(backend=backend)(make_data())
    assert graph._backend == backend
    if isinstance(backend, JaxBackend):
        assert isinstance(graph.hyper_edge_sets["bus"].feature_array, jnp.ndarray)


# ---------------------------------------------------------------------------
# Shape conventions: pad and collate
# ---------------------------------------------------------------------------


def test_pad_and_collate():
    """Graphs of different sizes must pad to a common shape and collate into a batch."""
    graphs = [DummyConverter()(make_data(n_bus=n)) for n in (3, 5)]

    numpy_backend = NumpyBackend()
    max_shape = GraphShape(
        backend=numpy_backend,
        hyper_edge_sets={"line": np.array(4), "bus": np.array(5)},
        addresses=np.array(5),
    )
    for graph in graphs:
        graph.pad(target_shape=max_shape)
    batch = collate_graphs(graphs)

    assert batch.hyper_edge_sets["bus"].feature_array.shape[:2] == (2, 5)
    assert batch.hyper_edge_sets["line"].port_dict["from"].shape == (2, 4)
    assert batch.non_fictitious_addresses.shape == (2, 5)


# ---------------------------------------------------------------------------
# _str_to_int
# ---------------------------------------------------------------------------


def test_str_to_int_deterministic_sorted_mapping():
    df = pd.DataFrame({"from": ["c", "a"], "to": ["b", "c"]})
    out, n_addresses = _str_to_int({"line": df})

    assert n_addresses == 3
    # sorted() order: a -> 0, b -> 1, c -> 2
    assert out["line"]["from"].tolist() == [2, 0]
    assert out["line"]["to"].tolist() == [1, 2]


def test_str_to_int_integer_addresses_identity():
    df = pd.DataFrame({"id": [0, 1, 2]})
    out, n_addresses = _str_to_int({"bus": df})

    assert n_addresses == 3
    assert out["bus"]["id"].tolist() == [0, 1, 2]


def test_str_to_int_none_table_passthrough():
    out, n_addresses = _str_to_int({"bus": pd.DataFrame({"id": ["x"]}), "global": None})

    assert out["global"] is None
    assert n_addresses == 1


# ---------------------------------------------------------------------------
# _any_to_float
# ---------------------------------------------------------------------------


def test_any_to_float_nan_and_inf():
    df = pd.DataFrame({"a": [np.nan, 1.0, np.inf, -np.inf, 1e12]})
    out = _any_to_float({"k": df})["k"]

    assert out["a"].tolist() == [0.0, 1.0, 1e6, -1e6, 1e6]


def test_any_to_float_categorical_deterministic():
    df = pd.DataFrame({"c1": ["open", "closed"], "c2": ["closed", "open"]})
    out = _any_to_float({"k": df})["k"]

    # Every value lands in [0, 1), and a given category gets the same float in both columns.
    assert ((out >= 0) & (out < 1)).all().all()
    assert out["c1"][0] == out["c2"][1]
    assert out["c1"][1] == out["c2"][0]
    assert out["c1"][0] != out["c1"][1]

    # And the same float across two separate calls.
    out_2 = _any_to_float({"k": pd.DataFrame({"c1": ["open"]})})["k"]
    assert out_2["c1"][0] == out["c1"][0]


def test_any_to_float_does_not_mutate_input():
    df = pd.DataFrame({"c": ["open", "closed"], "x": [np.nan, 1.0]})
    _any_to_float({"k": df})

    assert df["c"].dtype == object
    assert np.isnan(df["x"][0])


def test_any_to_float_none_table_passthrough():
    out = _any_to_float({"k": None})
    assert out["k"] is None
