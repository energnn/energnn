# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from energnn.amortizer import SimpleAmortizer
from energnn.amortizer.metadata import TrainerMetadata
from energnn.trainer_registry import TrainerRegistry


class DummyRegistry(TrainerRegistry):
    def register_trainer(self, trainer: SimpleAmortizer, best: bool = False, last: bool = False) -> bool:
        pass

    def get_trainer_metadata(self, run_id: str, step: int) -> TrainerMetadata | None:
        pass

    def download_trainer(self, run_id: str, step: int) -> SimpleAmortizer:
        pass
