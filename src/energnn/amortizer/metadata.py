# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#

class TrainerMetadata(dict):
    """Metadata of a SimpleAmortizer instance.

    :param name: Name of the instance.
    :param run_id: Identifier of the run in which the trainer was created.
    :param training_step: Step at which the trainer was stored.
    :param parent_run_id: If this run was started from another one, identifier of the run from which the trainer was created.
    :param best: Whether the trainer is the best one found for this run.
    :param last: Whether the trainer is the last one stored for this run.
    :param tags: Dictionary of tags associated with the trainer.
    """
    project_name: str
    name: str
    run_id: str
    training_step: int
    parent_run_id: str
    best: bool
    last: bool
    tags: dict
