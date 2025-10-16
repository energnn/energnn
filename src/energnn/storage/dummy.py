#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from .storage import Storage


class DummyStorage(Storage):

    def __init__(self):
        pass

    def upload(self, source_path: str, target_path: str) -> None:
        return None

    def download(self, source_path: str, target_path: str, overwrite: bool = False, unzip: bool = True) -> None:
        return None

    def delete(self, target_path: str) -> None:
        return None
