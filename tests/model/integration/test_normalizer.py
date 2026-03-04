# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.model import (
    CenterReduceNormalizer,
    TDigestNormalizer,
)
from tests.utils import TestProblemLoader

jax.config.update("jax_enable_x64", False)


def test_center_reduce_normalizer():
    loader = TestProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    normalizer = CenterReduceNormalizer(
        update_limit=1000,
        in_structure=loader.context_structure,
    )

    def f(normalizer, graph):
        return normalizer(graph=graph, get_info=False)

    coordinates_batch, _ = nnx.jit(f)(normalizer=normalizer, graph=context_batch)


def test_t_digest_normalizer():
    loader = TestProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    normalizer = TDigestNormalizer(in_structure=loader.context_structure, n_breakpoints=100, update_limit=100)

    def f(normalizer, graph):
        return normalizer(graph=graph, get_info=False)

    coordinates_batch, _ = nnx.jit(f)(normalizer=normalizer, graph=context_batch)
