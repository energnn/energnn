# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from omegaconf import DictConfig

from .tracker import Tracker


class DummyTracker(Tracker):
    """
    A no-op implementation of the Tracker interface.

    This dummy tracker provides empty implementations of all abstract methods,
    allowing experiment code to run without actual logging or external dependencies.
    Use this class for local development or unit tests where tracking is not required.
    """

    def __init__(self):
        pass

    def init_run(self, *, name: str, tags: list[str], cfg: DictConfig):
        return None

    def stop_run(self):
        return None

    def get_amortizer_path(self, *, tag: str) -> str:
        return ""

    def run_track_dataset(self, *, infos: dict, target_path: str) -> None:
        return None

    def run_track_amortizer(self, *, id: str, target_path: str) -> None:
        return None

    def run_append(self, *, infos: dict, step: int) -> None:
        return None
