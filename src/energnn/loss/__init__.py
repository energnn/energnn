# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from .losses import (
    L1Loss,
    MSELoss,
    SmoothL1Loss,
    HuberLoss,
    PoissonNLLLoss,
    BCELoss,
    BCEWithLogitsLoss,
    SoftMarginLoss,
    CrossEntropyLoss,
    NLLLoss,
    MultiMarginLoss,
    KLDivLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
)

__all__ = [
    "L1Loss",
    "MSELoss",
    "SmoothL1Loss",
    "HuberLoss",
    "PoissonNLLLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SoftMarginLoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "MultiMarginLoss",
    "KLDivLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
]
