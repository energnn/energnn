#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import chex
from unittest.mock import MagicMock
from flax.core.frozen_dict import unfreeze, freeze
import jax

from energnn.gnn.decoder import InvariantDecoder, EquivariantDecoder
from energnn.graph.jax import JaxGraph
from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.gnn.coupler.solving_method import SolvingMethod
from energnn.gnn.coupler import Coupler
import jax.numpy as jnp


def set_dense_layers_to_identity_or_zero(params, module_name, set_identity=True):
    """
    Patch params (Flax FrozenDict) such that Dense layers under `module_name` become:
      - identity kernel and zero bias if set_identity=True (square case),
      - zero kernel and zero bias if set_identity=False.

    Returns a new frozen params dict.
    """
    p = unfreeze(params)
    # typical structure: {'params': {module_name: {'Dense_0': {'kernel':..., 'bias':...}, ...}, ...}}
    if "params" not in p:
        raise KeyError("'params' key not found in params dict")
    top = p["params"]
    if module_name not in top:
        raise KeyError(f"Module '{module_name}' not found in params structure: {list(top.keys())}")
    mod = top[module_name]
    for layer_name, layer in list(mod.items()):
        if isinstance(layer, dict) and "kernel" in layer:
            k = np.array(layer["kernel"])
            b = np.array(layer.get("bias", np.zeros(k.shape[1], dtype=k.dtype)))
            in_dim, out_dim = k.shape
            if set_identity:
                new_k = np.zeros_like(k)
                for i in range(min(in_dim, out_dim)):
                    new_k[i, i] = 1.0
                new_b = np.zeros_like(b)
            else:
                new_k = np.zeros_like(k)
                new_b = np.zeros_like(b)
            mod[layer_name]["kernel"] = new_k.astype(np.float32)
            mod[layer_name]["bias"] = new_b.astype(np.float32)
    top[module_name] = mod
    p["params"] = top
    return freeze(p)

def make_dummy_coupling_mock():
    """Simple coupling stub that returns a fixed param on init and whose apply returns zeros of matching shape."""

    def init(*, rngs, context, coordinates):
        return {"dummy": 1}

    def init_with_output(*, rngs, context, coordinates):
        # return ((output, info), params)
        zeros = jnp.zeros_like(coordinates)
        return (zeros, {}), {"dummy": 1}

    def apply(params, context, coordinates, get_info=False):
        # returns zero update (same shape as coordinates) and info
        return jnp.zeros_like(coordinates), {"stub": jnp.array(1.0)}

    m = MagicMock(spec=CouplingFunction)
    m.init = init
    m.init_with_output = init_with_output
    m.apply = apply
    return m

def make_stub_solver_mock(coords_out):
    """Stub solver that records that it was called and returns a pre-defined coordinate/result."""

    def initialize_coordinates(*, context):
        # return zeros shaped array based on context
        return jnp.zeros([jnp.shape(context.non_fictitious_addresses)[0], coords_out.shape[1]])

    def solve(solver, *, params, function, context, coordinates_init, get_info=False):
        solver.called = True
        return coords_out, {"stub_solve": jnp.array(1.0)}

    m = MagicMock(spec=SolvingMethod)
    m.called = False
    m.initialize_coordinates = initialize_coordinates
    m.solve = lambda *, params, function, context, coordinates_init, get_info=False : solve(m, params=params,
                                                                                            function=function,
                                                                                            context=context,
                                                                                            coordinates_init=coordinates_init,
                                                                                            get_info=get_info)
    return m

def assert_single(*, coupler: Coupler, seed: int, context: JaxGraph):
    rngs = jax.random.PRNGKey(seed)
    params_1 = coupler.init(context=context, rngs=rngs)
    output_3, infos_3 = coupler.apply(params_1, context=context)
    output_4, infos_4 = coupler.apply(params_1, context=context, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)

    return params_1, output_4, infos_4


def assert_batch(*, params: dict, coupler: Coupler, context: JaxGraph):

    def apply(params, context, get_info):
        return coupler.apply(params, context=context, get_info=get_info)

    apply_vmap = jax.vmap(apply, in_axes=[None, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, context, False)
    output_batch_2, infos_2 = apply_vmap(params, context, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, context, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, context, True)

    chex.assert_trees_all_close(output_batch_1, output_batch_2, output_batch_3, output_batch_4, rtol=1e-4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    return output_batch_1, infos_1