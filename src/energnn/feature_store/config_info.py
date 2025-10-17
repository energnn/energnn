# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import uuid


class ProblemGenerationConfigInfo(dict):
    """
    Represents metadata associated with a problem generation configuration.

    Keys expected in this dictionary include:

    :param id: Unique configuration identifier.
    :param hash: MD5 or other hash of the configuration file to ensure integrity.
    :param tags: Optional key-value metadata tags for categorization or filtering.
    :param storage_path: Unique path or reference used in remote storage.
    """

    id: str
    hash: str
    tags: dict = {}
    storage_path: uuid.UUID
