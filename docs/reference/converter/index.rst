=========
Converter
=========

Converters turn domain-specific objects (e.g. a ``pypowsybl.network.Network``) into
:class:`~energnn.graph.Graph` instances.

A :class:`~energnn.converter.Converter` is composed of one
:class:`~energnn.converter.ElementsConverter` per hyper-edge class (e.g. ``"bus"``, ``"line"``),
each in charge of extracting a table of addresses and features from the input object.
The converter then maps string addresses to consecutive integers, casts features to bounded
floats, and assembles the tables into a graph.

.. currentmodule:: energnn.converter


Converter
=========

.. autoclass:: Converter
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Converter.__call__
   Converter.get_structure


ElementsConverter
=================

.. autoclass:: ElementsConverter
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ElementsConverter.__init__
   ElementsConverter.__call__
   ElementsConverter.get_structure