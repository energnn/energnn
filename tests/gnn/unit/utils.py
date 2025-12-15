import numpy as np
import chex
from flax.core.frozen_dict import unfreeze, freeze
import jax

from energnn.gnn.decoder import InvariantDecoder, EquivariantDecoder
from energnn.graph.jax import JaxGraph


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