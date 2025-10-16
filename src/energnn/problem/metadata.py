#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
class ProblemMetadata(dict):
    """Metadata of a Problem instance.

    :param name: Name of the instance.
    :param config_id: Identifier of the configuration file used to generate the instance.
    :param code_version: Version of the code used to generate the instance.
    :param context_shape: Shape of the context of the instance, formatted as a dict containing only int values
        (no jax.Array).
    :param decision_shape: Shape of the decision of the instance, formatted as a dict containing only int values
        (no jax.Array).
    :param filter_tags: Dictionary of criteria used to select the instance to form datasets.
    """

    def __init__(
        self,
        name: str,
        config_id: str,
        code_version: int,
        context_shape: dict,
        decision_shape: dict,
        storage_path: str = "",
        filter_tags: dict | None = None,
    ) -> None:
        super().__init__()
        if filter_tags is None:
            filter_tags = {}
        self["name"] = name
        self["config_id"] = config_id
        self["code_version"] = code_version
        self["context_shape"] = context_shape
        self["decision_shape"] = decision_shape
        self["storage_path"] = storage_path
        self["filter_tags"] = filter_tags

    @property
    def name(self):
        return self["name"]

    @property
    def config_id(self):
        return self["config_id"]

    @property
    def code_version(self):
        return self["code_version"]

    @property
    def context_shape(self):
        return self["context_shape"]

    @property
    def decision_shape(self):
        return self["decision_shape"]

    @property
    def filter_tags(self):
        return self["filter_tags"]

    @property
    def storage_path(self):
        return self["storage_path"]
