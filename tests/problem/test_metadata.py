#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import json
import pickle

import numpy as np
import pytest

from energnn.problem.metadata import ProblemMetadata


@pytest.fixture
def sample_metadata():
    return ProblemMetadata(
        name="instance_001",
        config_id="cfg_v1",
        code_version=42,
        context_shape={"node": 10, "edge": 5},
        decision_shape={"node": 10},
    )


def test_init_stores_values(sample_metadata):
    m = sample_metadata
    # direct dict access
    assert m["name"] == "instance_001"
    assert m["config_id"] == "cfg_v1"
    assert m["code_version"] == 42
    assert m["context_shape"] == {"node": 10, "edge": 5}
    assert m["decision_shape"] == {"node": 10}

    # properties
    assert m.name == "instance_001"
    assert m.config_id == "cfg_v1"
    assert m.code_version == 42
    assert m.context_shape == {"node": 10, "edge": 5}
    assert m.decision_shape == {"node": 10}

    # defaults
    assert m.storage_path == ""
    assert m.filter_tags == {}


def test_defaults_filter_tags_and_storage_path():
    m = ProblemMetadata(
        name="n",
        config_id="c",
        code_version=1,
        context_shape={},
        decision_shape={},
    )
    assert m.storage_path == ""
    assert m.filter_tags == {}


def test_property_reflects_underlying_dict():
    m = ProblemMetadata(
        name="foo",
        config_id="bar",
        code_version=7,
        context_shape={"a": 1},
        decision_shape={"b": 2},
    )
    # modify underlying dict directly
    m["name"] = "changed"
    assert m.name == "changed"

    # modify nested structure and ensure property sees change (reference semantics)
    m["context_shape"]["a"] = 99
    assert m.context_shape["a"] == 99


def test_mapping_behaviour_and_mutability():
    m = ProblemMetadata(
        name="x",
        config_id="y",
        code_version=0,
        context_shape={"k": 1},
        decision_shape={"d": 2},
    )
    # add a new key (dict-like)
    m["new_key"] = 123
    assert m["new_key"] == 123

    # delete a key and accessing property should raise KeyError
    del m["config_id"]
    with pytest.raises(KeyError):
        _ = m.config_id


def test_dict_and_pickle_serialization(sample_metadata):
    m = sample_metadata
    d = dict(m)
    # dict conversion preserves values
    assert d["name"] == m.name
    assert d["context_shape"] == m.context_shape

    # pickle roundtrip
    packed = pickle.dumps(m)
    loaded = pickle.loads(packed)
    assert isinstance(loaded, ProblemMetadata)
    assert dict(loaded) == dict(m)

    # json serialization of dict (should succeed for simple types)
    json_str = json.dumps(dict(m))
    assert isinstance(json_str, str)
    reloaded = json.loads(json_str)
    # json loads produce strings/numbers/dicts - compare relevant keys
    assert reloaded["name"] == m.name
    assert int(reloaded["code_version"]) == m.code_version


def test_no_type_validation_context_decision_shape():
    # Pass unusual types (numpy array) inside shapes and ensure stored as-is
    ctx = {"node": np.array([1, 2, 3])}
    dec = {"out": np.array([4])}
    m = ProblemMetadata(name="t", config_id="c", code_version=2, context_shape=ctx, decision_shape=dec)
    # The class does not validate types -> stored as provided
    assert isinstance(m.context_shape["node"], np.ndarray)
    assert np.array_equal(m.context_shape["node"], np.array([1, 2, 3]))


def test_empty_shapes_supported():
    m = ProblemMetadata(name="e", config_id="c", code_version=3, context_shape={}, decision_shape={})
    assert m.context_shape == {}
    assert m.decision_shape == {}


def test_custom_filter_tags_stored():
    tags = {"difficulty": "hard", "seed": 123}
    m = ProblemMetadata(
        name="tagged",
        config_id="cfg",
        code_version=9,
        context_shape={"a": 1},
        decision_shape={"b": 2},
        filter_tags=tags,
    )
    # stored by reference: modifying original dict will reflect in the metadata (documented behavior)
    assert m.filter_tags == tags
    tags["difficulty"] = "easy"
    assert m.filter_tags["difficulty"] == "easy"


def test_properties_return_reference_not_copy():
    m = ProblemMetadata(
        name="ref",
        config_id="cfg",
        code_version=11,
        context_shape={"node": 1},
        decision_shape={"edge": 2},
    )
    # properties should expose same object (not copy) -> modification visible via dict and property
    prop = m.context_shape
    prop["node"] = 999
    assert m["context_shape"]["node"] == 999
    assert m.context_shape["node"] == 999


def test_equality_by_dict_contents():
    a = ProblemMetadata(
        name="same",
        config_id="c1",
        code_version=5,
        context_shape={"x": 1},
        decision_shape={"y": 2},
    )
    b = ProblemMetadata(
        name="same",
        config_id="c1",
        code_version=5,
        context_shape={"x": 1},
        decision_shape={"y": 2},
    )
    # equality by dict contents (not by object identity)
    assert dict(a) == dict(b)
