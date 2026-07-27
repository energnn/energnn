# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import pandas as pd

from energnn.graph import HyperEdgeSetStructure


class ElementsConverter(ABC):
    """Abstract base class for elements converters.

    An elements converter extracts, for one class of hyper-edges (e.g. ``"bus"``, ``"line"``),
    a table of addresses and a table of features from a domain-specific object. Subclasses must
    implement :meth:`_get_table`, which returns a single :class:`pandas.DataFrame` containing at
    least the columns listed in ``port_list`` and ``feature_list``; :meth:`__call__` then splits
    it into the ``(address table, feature table)`` pair consumed by :class:`Converter`.

    At least one of ``port_list`` and ``feature_list`` must be provided: a hyper-edge set with
    neither ports nor features would be empty.

    :param port_list: Names of the columns of the table returned by :meth:`_get_table` that
        contain addresses (e.g. ``["from", "to"]`` for a line). ``None`` if the hyper-edges have
        no ports (e.g. global quantities).
    :param feature_list: Names of the columns that contain features. ``None`` if the hyper-edges
        carry no feature (e.g. purely structural elements).

    :ivar attributes: All columns that must be fetched from the underlying data source, i.e. the
        concatenation of ``port_list`` and ``feature_list``.
    """

    def __init__(self, port_list: list[str] | None, feature_list: list[str] | None):
        if port_list is None and feature_list is None:
            raise ValueError("At least one of port_list and feature_list must be provided.")

        self.port_list = port_list
        self.feature_list = feature_list

        self.attributes = []
        if self.port_list is not None:
            self.attributes.extend([s for s in self.port_list])
        if self.feature_list is not None:
            self.attributes.extend(self.feature_list)

    def __call__(self, *args, **kwargs) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Extract the ``(address table, feature table)`` pair for this class of hyper-edges.

        All arguments are forwarded verbatim to :meth:`_get_table`.

        :return: The address table (columns ``port_list``, or ``None``) and the feature table
            (columns ``feature_list``, or ``None``); both share the same row order, one row per
            hyper-edge.
        """
        df = self._get_table(*args, **kwargs)

        if self.port_list is not None:
            df_port = df[self.port_list]
        else:
            df_port = None

        if self.feature_list is not None:
            df_feature = df[self.feature_list]
        else:
            df_feature = None

        return df_port, df_feature

    @abstractmethod
    def _get_table(self, *args, **kwargs) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` containing addresses and features.

        The DataFrame must have one row per hyper-edge and contain at least the columns listed
        in ``port_list`` and ``feature_list``.
        """
        raise NotImplementedError

    def get_structure(self) -> HyperEdgeSetStructure:
        """Get the edge structure of the element, useful for building an EnerGNN model."""
        return HyperEdgeSetStructure(port_list=self.port_list, feature_list=self.feature_list)
