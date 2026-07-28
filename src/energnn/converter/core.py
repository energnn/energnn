# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import random

import numpy as np
import pandas as pd

from energnn.graph import Graph, GraphStructure, HyperEdgeSet
from energnn.graph.backend import Backend, NumpyBackend
from .element_converter import ElementsConverter


class Converter:
    """Abstract base class for all converters.

    A converter turns a domain-specific object (e.g. a ``pypowsybl.network.Network``) into an
    :class:`energnn.graph.Graph`. Subclasses only need to provide ``elements_converter_dict``,
    which maps each hyper-edge class name (e.g. ``"bus"``, ``"line"``) to the
    :class:`ElementsConverter` in charge of extracting its table of addresses and features.

    Calling the converter runs the following pipeline:

    1. each :class:`ElementsConverter` extracts an ``(address table, feature table)`` pair from
       the input object;
    2. string addresses are mapped to consecutive integers ``0..n-1``, in sorted order so that
       the mapping is reproducible across runs;
    3. features are cast to floats: categorical values are hashed to a deterministic float in
       ``[0, 1)``, NaNs are replaced by 0, and values are clipped to ``[-1e6, 1e6]``;
    4. the tables are assembled into a :class:`energnn.graph.Graph` through
       :meth:`HyperEdgeSet.from_dict` and :meth:`Graph.from_dict`.

    The graph is always built on a numpy backend, because graph shapes vary from one input to
    the next and building directly in jax would trigger one XLA compilation per new shape. If
    ``backend`` is set, the finished graph is converted to it with :meth:`Graph.to_backend`.

    :cvar elements_converter_dict: Mapping from hyper-edge class name to its
        :class:`ElementsConverter`. Must be defined by subclasses.
    :cvar backend: Optional target backend for the returned graph. If ``None``, the graph is
        returned on a :class:`NumpyBackend`.
    """

    elements_converter_dict: dict[str, ElementsConverter]
    backend: Backend | None = None

    def get_structure(self) -> GraphStructure:
        """Return the :class:`GraphStructure` describing the graphs produced by this converter.

        Useful for building an EnerGNN model without having to convert an actual input first.
        """
        return GraphStructure(hyper_edge_sets={k: c.get_structure() for k, c in self.elements_converter_dict.items()})

    def __call__(self, *args, **kwargs) -> Graph:
        """Convert an input object into a :class:`energnn.graph.Graph`.

        All arguments are forwarded verbatim to each :class:`ElementsConverter`.

        :return: A graph on a numpy backend, or on ``self.backend`` if it is set.
        """
        # Build dict of tables
        tables = {}
        for k, element_converter in self.elements_converter_dict.items():
            tables[k] = element_converter(*args, **kwargs)

        # First, convert str addresses into unique integers.
        df_port_dict = {k: tables[k][0] for k in tables.keys()}
        df_port_int_dict, n_addresses = _str_to_int(df_port_dict)

        # Then, convert features into floats.
        df_feature_dict = {k: tables[k][1] for k in tables.keys()}
        df_feature_float_dict = _any_to_float(df_feature_dict)

        # Convert tables into an energnn.graph.Graph.
        tables = {k: (df_port_int_dict[k], df_feature_float_dict[k]) for k in tables.keys()}
        graph = _tables_to_graph(backend=self.backend, tables=tables, n_addresses=n_addresses)

        return graph


def _str_to_int(df_port_dict: dict[str, pd.DataFrame | None]) -> tuple[dict[str, pd.DataFrame | None], int]:
    """Convert addresses into unique consecutive integers.

    Addresses are enumerated in sorted order so that the mapping only depends on the set of
    addresses, not on hash randomization or on the order of the tables.

    :param df_port_dict: Mapping from hyper-edge class name to its address table (or ``None``).
    :return: The translated tables and the total number of distinct addresses.
    """

    # 1. Gather the sorted array of all distinct addresses. The distinct values are collected
    #    with a hash table (fast on string data) so that only the uniques need to be sorted.
    address_arrays = [df.values.ravel() for df in df_port_dict.values() if df is not None]
    if not address_arrays:
        return dict.fromkeys(df_port_dict), 0
    all_addresses = pd.Index(np.sort(pd.unique(np.concatenate(address_arrays))))

    # 2. Translate each table through a vectorized hash lookup against the sorted addresses.
    out_dict = {}
    for k, df in df_port_dict.items():
        if df is not None:
            indices = all_addresses.get_indexer(df.values.ravel()).reshape(df.shape)
            out_dict[k] = pd.DataFrame(indices, columns=df.columns, index=df.index)
        else:
            out_dict[k] = None

    return out_dict, len(all_addresses)


def _any_to_float(
    feature_tables: dict[str, pd.DataFrame | None],
    min_val: float = -1e6,
    max_val: float = 1e6,
) -> dict[str, pd.DataFrame | None]:
    """Convert feature tables to bounded float tables, without mutating the inputs.

    Categorical (string/object) columns are mapped value-wise to a float in ``[0, 1)`` by
    seeding a PRNG with the value, so that a given category always gets the same float, across
    columns and across runs. NaNs are replaced by 0, and ±inf and out-of-range values are
    clipped to ``[min_val, max_val]``.

    :param feature_tables: Mapping from hyper-edge class name to its feature table (or ``None``).
    :param min_val: Lower bound applied to the output values.
    :param max_val: Upper bound applied to the output values.
    :return: New float tables, in the same layout as the input.
    """

    out_dict = {}
    for k, df in feature_tables.items():
        if df is not None:
            df = df.copy()

            # Convert categorical features into floats: hash each distinct category once, then
            # broadcast to the rows (there are usually far fewer categories than rows).
            for col in df.columns:
                if df[col].dtype.kind in {"U", "S", "O"}:
                    hash_map = {value: random.Random(value).random() for value in df[col].dropna().unique()}
                    df[col] = df[col].map(hash_map)

            df = df.astype(float)
            df = df.replace([-np.inf, np.inf], [min_val, max_val])
            out_dict[k] = df.fillna(0).clip(min_val, max_val)
        else:
            out_dict[k] = None
    return out_dict


def _tables_to_graph(
    *,
    backend: Backend | None = None,
    tables: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]],
    n_addresses: int,
) -> Graph:
    """Assemble translated tables into a :class:`energnn.graph.Graph`.

    The graph is built on a numpy backend (variable shapes are free there) and converted to
    ``backend`` at the very end, if one is given.

    :param backend: Optional target backend for the returned graph.
    :param tables: Mapping from hyper-edge class name to its ``(address table, feature table)``
        pair, as produced by :func:`_str_to_int` and :func:`_any_to_float`.
    :param n_addresses: Total number of distinct addresses in the graph.
    :return: The assembled graph.
    """

    numpy_backend = NumpyBackend()

    hyper_edge_set_dict = {}
    for k, (df_address, df_feature) in tables.items():
        if df_address is not None:
            port_dict = {kk: df_address[kk].to_numpy(dtype=np.int32) for kk in df_address.columns}
        else:
            port_dict = None

        if df_feature is not None:
            feature_dict = {kk: df_feature[kk].to_numpy(dtype=np.float64) for kk in df_feature.columns}
        else:
            feature_dict = None

        hyper_edge_set_dict[k] = HyperEdgeSet.from_dict(backend=numpy_backend, port_dict=port_dict, feature_dict=feature_dict)

    graph = Graph.from_dict(backend=numpy_backend, hyper_edge_set_dict=hyper_edge_set_dict, n_addresses=np.array(n_addresses))

    if backend is not None and not isinstance(backend, NumpyBackend):
        graph = graph.to_backend(backend)
    return graph
