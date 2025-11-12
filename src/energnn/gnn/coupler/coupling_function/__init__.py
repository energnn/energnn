# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from .coupling_function import CouplingFunction
from .local_message_function import (
    LocalMessageFunction,
    EmptyLocalMessageFunction,
    IdentityLocalMessageFunction,
    SumLocalMessageFunction,
)
from .remote_message_function import (
    RemoteMessageFunction,
    EmptyRemoteMessageFunction,
    IdentityRemoteMessageFunction,
)
from .self_message_function import (
    SelfMessageFunction,
    EmptySelfMessageFunction,
    IdentitySelfMessageFunction,
    MLPSelfMessageFunction,
)

__all__ = [
    "CouplingFunction",
    "LocalMessageFunction",
    "EmptyLocalMessageFunction",
    "IdentityLocalMessageFunction",
    # "GATv2LocalMessageFunction",
    "SumLocalMessageFunction",
    "SelfMessageFunction",
    "EmptySelfMessageFunction",
    "IdentitySelfMessageFunction",
    "MLPSelfMessageFunction",
    "RemoteMessageFunction",
    "EmptyRemoteMessageFunction",
    "IdentityRemoteMessageFunction",
]
