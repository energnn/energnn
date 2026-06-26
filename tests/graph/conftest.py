# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest

from energnn.graph.backend import JaxBackend, NumpyBackend


@pytest.fixture(params=["numpy", "jax"], ids=["numpy", "jax"])
def backend(request):
    """Parametrize tests over both array backends."""
    if request.param == "numpy":
        return NumpyBackend()
    return JaxBackend()
